#!/usr/bin/env python
"""Reassemble an HDF5 file that was split with split_hdf5.py.

Point this at the directory holding the downloaded shards (or list the shard
files explicitly) and it rebuilds the original single .hdf5 file.

The shards carry a manifest in their root attributes, so the script checks that
the set of shards is complete and consistent before writing anything.  Unrelated
files in the same folder (.pkl, .json, README, subdirectories, or an .hdf5 from a
different dataset) are ignored, so you can point this straight at a download
directory.  Files written by something else can still be merged on purpose with
--no-manifest.

Usage:
    python merge_hdf5.py shards/ -o iclr_real_data.hdf5
    python merge_hdf5.py shards/*.hdf5 -o iclr_real_data.hdf5
    python merge_hdf5.py shards/ -o out.hdf5 --dry-run
"""

import argparse
import glob
import json
import os
import sys
import time

import h5py

MANIFEST_PREFIX = "__split__"


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def as_str(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def collect_shards(inputs, output):
    """Expand directories / globs into a sorted list of (path, was_scanned).

    was_scanned is True for files found by scanning a directory.  Those are
    only candidates -- anything without a split manifest is skipped with a
    warning, so unrelated files in the folder (a stray .hdf5, a previous merge
    result) do not derail the merge.  Files the user named explicitly are taken
    at their word and must be real shards.

    The scan is deliberately non-recursive and limited to *.hdf5 / *.h5, so
    .pkl / .json / README files and subdirectories are ignored outright.
    """
    candidates = []
    for item in inputs:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, "*.hdf5")))
            found += sorted(glob.glob(os.path.join(item, "*.h5")))
            if not found:
                sys.exit(f"No .hdf5/.h5 files found in directory: {item}")
            candidates.extend((p, True) for p in found)
        else:
            expanded = sorted(glob.glob(item))
            if not expanded:
                sys.exit(f"No such file: {item}")
            candidates.extend((p, False) for p in expanded)

    # De-duplicate (keeping the strictest origin) and never treat the file we
    # are about to write as an input.
    seen, unique = {}, []
    for path, scanned in candidates:
        real = os.path.abspath(path)
        if real == output:
            continue
        if real in seen:
            if not scanned:
                seen[real][1] = False
            continue
        entry = [real, scanned]
        seen[real] = entry
        unique.append(entry)
    if not unique:
        sys.exit("No candidate shard files left to merge.")
    return [(p, s) for p, s in unique]


def read_manifest(path):
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        manifest = {k[len(MANIFEST_PREFIX):]: v
                    for k, v in attrs.items() if k.startswith(MANIFEST_PREFIX)}
        root_attrs = {k: v for k, v in attrs.items()
                      if not k.startswith(MANIFEST_PREFIX)}
        top_level = sorted(f.keys())
    return manifest, root_attrs, top_level


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Shard files, glob patterns, or a directory containing them")
    ap.add_argument("-o", "--output", required=True,
                    help="Path of the merged .hdf5 file to write")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output file if it already exists")
    ap.add_argument("--no-manifest", action="store_true",
                    help="Merge in filename order without checking the split manifest")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate the shards and print the plan without writing")
    args = ap.parse_args()

    output = os.path.abspath(args.output)
    shards = collect_shards(args.inputs, output)
    if os.path.exists(output) and not args.overwrite and not args.dry_run:
        sys.exit(f"Output already exists (use --overwrite): {output}")

    print(f"Found {len(shards)} candidate file(s):")
    ordered = []          # (path, top_level_names)
    root_attrs = {}
    expected_total = None

    if args.no_manifest:
        for path, _ in shards:
            _, attrs, names = read_manifest(path)
            root_attrs.update(attrs)
            ordered.append((path, names))
            print(f"  {os.path.basename(path):>50s}  {len(names):4d} objects  "
                  f"{human(os.path.getsize(path)):>10s}")
    else:
        by_index = {}
        num_parts_seen = set()
        skipped = []
        for path, scanned in shards:
            manifest, attrs, names = read_manifest(path)
            if "part_index" not in manifest:
                if scanned:
                    # Not one of ours -- some other .hdf5 that happens to live in
                    # the same folder.  Leave it alone; if a genuine shard were
                    # skipped here, the completeness check below still catches it.
                    skipped.append(path)
                    continue
                sys.exit(f"{path} has no split manifest, so it was not produced by "
                         f"split_hdf5.py. Remove it from the list, or pass "
                         f"--no-manifest to merge its contents in regardless.")
            index = int(manifest["part_index"])
            num_parts_seen.add(int(manifest["num_parts"]))
            if expected_total is None:
                expected_total = int(manifest["total_objects"])
            if index in by_index:
                sys.exit(f"Duplicate shard for part {index + 1}:\n"
                         f"  {by_index[index][0]}\n  {path}")
            listed = json.loads(as_str(manifest["object_names"]))
            if sorted(listed) != names:
                sys.exit(f"{path}: contents do not match its manifest "
                         f"({len(names)} objects present, {len(listed)} listed). "
                         f"The file is probably truncated or corrupted — re-download it.")
            by_index[index] = (path, listed)
            root_attrs.update(attrs)

        for path in skipped:
            print(f"  skipping {os.path.basename(path)} (not a shard of this split)")
        if not by_index:
            sys.exit("None of these files carry a split manifest, so none of them "
                     "were produced by split_hdf5.py. Check that you are pointing "
                     "at the downloaded shards.")
        if len(num_parts_seen) != 1:
            sys.exit(f"Shards disagree on the total number of parts: "
                     f"{sorted(num_parts_seen)}")
        num_parts = num_parts_seen.pop()
        missing = [i + 1 for i in range(num_parts) if i not in by_index]
        if missing:
            sys.exit(f"Missing shard(s) {missing} of {num_parts}. "
                     f"Download all {num_parts} parts before merging.")

        for i in range(num_parts):
            path, names = by_index[i]
            ordered.append((path, names))
            print(f"  part {i + 1}/{num_parts}  {os.path.basename(path):>50s}  "
                  f"{len(names):4d} objects  {human(os.path.getsize(path)):>10s}")

    all_names = [n for _, names in ordered for n in names]
    duplicates = sorted({n for n in all_names if all_names.count(n) > 1})
    if duplicates:
        sys.exit(f"The shards contain {len(duplicates)} duplicated top-level "
                 f"object(s), e.g. {duplicates[:5]}")
    if expected_total is not None and len(all_names) != expected_total:
        sys.exit(f"Object count mismatch: shards hold {len(all_names)} objects "
                 f"but the original had {expected_total}.")

    print(f"\nTotal: {len(all_names)} top-level objects -> {output}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return

    started = time.time()
    tmp = output + ".incomplete"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with h5py.File(tmp, "w") as dst:
        for key, value in root_attrs.items():
            dst.attrs[key] = value
        done = 0
        for path, names in ordered:
            print(f"[{os.path.basename(path)}] copying {len(names)} objects", flush=True)
            with h5py.File(path, "r") as src:
                for name in names:
                    src.copy(name, dst, name=name)
                    done += 1
                    print(f"    [{done}/{len(all_names)}] {name}", flush=True)
    os.replace(tmp, output)

    print(f"\nMerged into {output} ({human(os.path.getsize(output))}) "
          f"in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
