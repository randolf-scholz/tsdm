# Building documentation

Run these commands from the repository root:

- `just docs::build` (or `just docs::make`) builds the Zensical site in
  `docs/build/site` with strict validation.
- `just docs::build --clean` clears the Zensical cache before building.
- `just docs::clean` removes `docs/build`, including staged sources and the site.

From this directory, omit the `docs::` prefix. The build commands accept extra
arguments for Zensical.

Authored pages and their attachments live in `content/`. The build runs
`prepare_sources.py`, which copies that directory recursively into `build/sources/`,
stages theme assets, and generates API reference Markdown and `reference/SUMMARY.md`
navigation from the Python source. It reads the staging path from `docs_dir` in
`zensical.toml`. Each run replaces the staged sources so renamed or deleted pages
do not linger. Zensical renders those sources into `build/site/` and reuses its
cache during normal builds.

Site appearance and rendering live in `theme/`: HTML templates in `templates/`,
stylesheets in `css/`, and MathJax configuration directly in `mathjax_config.js`.
The build stages stylesheets under `build/sources/assets/css/` and the MathJax
configuration at `build/sources/assets/mathjax_config.js`. Zensical loads templates
directly from `theme/templates/`; they are not copied into the site sources.

To publish an image or downloadable document, place it under `content/assets/`
and link it from a page. This README contains maintainer instructions and is not
copied into the site.
