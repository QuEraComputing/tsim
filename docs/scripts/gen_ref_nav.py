# type: ignore
"""Generate the API reference pages for Tsim's public and experimental API."""

import mkdocs_gen_files

PAGES = {
    "index.md": (
        "Tsim API",
        "The public API, which follows semantic versioning. Other modules are "
        "internal and may change without notice.",
        [
            "tsim.Circuit",
            "tsim.CompiledMeasurementSampler",
            "tsim.CompiledDetectorSampler",
        ],
    ),
    "experimental.md": (
        "Experimental",
        "Helpers outside of the stable API. They may change or be removed in any "
        "release.",
        [
            "tsim.experimental.CompiledStateProbs",
            "tsim.experimental.SteaneEncoder",
            "tsim.experimental.ColorEncoder5",
        ],
    ),
}

nav = mkdocs_gen_files.Nav()
for doc_path, (title, intro, objects) in PAGES.items():
    nav[title] = doc_path
    with mkdocs_gen_files.open(f"reference/{doc_path}", "w") as fd:
        fd.write(f"# {title}\n\n{intro}\n\n")
        fd.writelines(f"::: {obj}\n\n" for obj in objects)

with mkdocs_gen_files.open("reference/SUMMARY.txt", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
