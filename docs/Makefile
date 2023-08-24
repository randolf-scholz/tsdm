# Minimal makefile for Sphinx documentation
#
export PYTORCH_JIT=0

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# create build directory if not exists
	mkdir -p _build

	# backup previous build
	mkdir -p _previous_build
	rm -rf _previous_build
	mv _build _previous_build

	# remove previous build
	rm -rf _build
	rm -rf _autosummary
	rm -rf autoapi
	rm -rf apidoc
	rm -rf api

	sphinx-apidoc  --separate  --implicit-namespaces  -o apidoc/  ../src/tsdm
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
