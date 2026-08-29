# Release and archival checklist

This checklist keeps the GitHub release, PyPI package, Zenodo archive, and JOSS manuscript aligned.

## One-time Zenodo setup

1. Sign in to Zenodo and connect the GitHub account that administers
   `tacular-omics/spxtacular`.
2. In Zenodo's GitHub settings, synchronize repositories and enable `spxtacular`.
3. Confirm that Zenodo recognizes the root `CITATION.cff` metadata.

Zenodo will archive each subsequent GitHub release after the integration is enabled. Do not also
upload the same release manually, because that would create a duplicate record.

## Prepare a release

1. Work from a clean branch based on the current default branch.
2. Set the same version in `src/spxtacular/__init__.py`, `CITATION.cff`, and the heading in
   `HISTORY.md`.
3. Confirm that the changelog date and citation metadata are correct.
4. Confirm that mzMLPy 0.7 or later is available from PyPI, then refresh and commit the tested lock with `uv lock --upgrade-package mzmlpy`.
5. Run:

   ```bash
   just lint
   just fmt-check
   just check
   just test-cov
   just docs-build
   just paper-preflight
   uv build
   ```

6. Inspect the wheel and source archive under `dist/`. Install the wheel in a clean environment for
   a final import smoke test.
7. Merge the release branch to `main` and wait for all required checks to pass.

## Publish and archive

1. Create a GitHub release whose tag is `vX.Y.Z` and whose target is the reviewed commit on `main`.
   Use the matching `HISTORY.md` section as the release notes.
2. The release workflow publishes the distributions to PyPI. Confirm the version and install it in
   a clean environment.
3. Wait for Zenodo to finish processing the GitHub release. Verify the record title, creator and
   ORCID, version, MIT license, keywords, repository link, files, and archival status before using
   the DOI in citations.
4. Add the resulting Zenodo DOI badge and citation text to `README.md` in a follow-up commit. Cite a
   version-specific DOI when exact reproducibility matters.
5. For JOSS, follow the editor's timing: the final tagged release and archive DOI are normally
   reported at the end of review. Add the accepted JOSS article DOI to the software record as a
   related identifier once it exists.

Publishing a Zenodo record is irreversible in the ordinary workflow. Preview its files and metadata
before confirming publication.
