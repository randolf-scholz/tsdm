# Building documentation

Run these commands from the repository root:

- `just docs::build` (or `just docs::make`) builds the Zensical site in
  `docs/build/site` with strict validation.
- `just docs::build --clean` clears the Zensical cache before building.
- `just docs::clean` removes `docs/build`, including staged sources and the site.

From this directory, omit the `docs::` prefix. The build commands accept extra
arguments for Zensical.

Published pages and assets live in `content/`. The build copies that directory
recursively into `build/sources/`, then adds generated API pages and navigation.
The generator reads the staging path from `docs_dir` in `zensical.toml`. Each run
replaces the staged sources so renamed or deleted pages do not linger. Zensical
renders those sources into `build/site/` and reuses its cache during normal builds.

Theme templates live in `overrides/`, and build tooling lives in `scripts/`.
Unpublished research material, notebooks, and diagram sources live in `notes/`.
To publish an image or downloadable document, place it under `content/assets/`
and link it from a page. This README contains maintainer instructions and is not
copied into the site.
