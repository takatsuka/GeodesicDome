# Releasing mtGeodesicDome

Releases are built and uploaded to [PyPI](https://pypi.org/project/mtgeodesicdome/) by GitHub
Actions (`.github/workflows/publish.yml`) when you publish a GitHub release. PyPI "trusted publishing" is
used, so no API token or password is stored anywhere.

## One-time set-up

1. **GitHub environments.** Under Settings → Environments of `takatsuka/mtGeodesicDome`, create `pypi`
   and `testpypi`. For `pypi` you can add yourself as a *required reviewer*, so every upload waits for a click.
2. **PyPI trusted publisher.** At <https://pypi.org>, go to *Your account → Publishing → Add a new pending
   publisher → GitHub*: project `mtgeodesicdome`, owner `takatsuka`, repository `mtGeodesicDome`,
   workflow `publish.yml`, environment `pypi`.
3. **TestPyPI (for rehearsals).** Same at <https://test.pypi.org> (a separate account), environment `testpypi`.
4. **Zenodo DOI (recommended).** Log in to <https://zenodo.org> with GitHub and switch the repository on.
   Each GitHub release then gets its own DOI; put the concept-DOI badge in the README.

## Making a release

1. Set the version in **two** places: `src/mt/geodesicdome/__init__.py` (`__version__`) and
   `CITATION.cff` (`version:` and `date-released:`). Move the *Unreleased* entries in `CHANGELOG.md`
   under a new heading for this version.
2. Run `pytest` and `ruff check .`. Optionally `python -m build && python -m twine check dist/*`.
3. Commit and push; check the **tests** workflow is green.
4. *Optional rehearsal:* Actions → publish → *Run workflow* uploads to TestPyPI. Try it with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mtgeodesicdome`.
5. **Publish:** Releases → *Draft a new release*, tag `v<version>` (e.g. `v1.2.0`), *Publish release*.
   The workflow checks the tag matches both version numbers, builds, tests the wheel in a clean
   environment and uploads to PyPI.

## First release (1.2.0) checklist

1. `mv _github .github` if the folder is still named `_github`, then commit and push everything to `main`,
   and make the repository **public**. PyPI shows the README's images from GitHub, so they appear only once
   `main` is public.
2. Do the *One-time set-up* above (the `pypi`/`testpypi` environments and the PyPI **pending** publisher for
   project `mtgeodesicdome`).
3. Optional: rehearse on TestPyPI (step 4 above).
4. Publish the GitHub release with tag `v1.2.0`. The workflow creates the `mtgeodesicdome` project on PyPI.

**Manual upload (fallback, without GitHub Actions).** Create an API token at <https://pypi.org/manage/account/token/>, then:

```bash
rm -rf dist && python -m build && python -m twine check --strict dist/*
python -m twine upload dist/*          # user name: __token__   password: the token
```

PyPI never accepts the same version twice. If a release is broken, *yank* it on PyPI and publish a new version.
