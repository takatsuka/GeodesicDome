# Contributing to GeodesicDome

Thank you for considering a contribution! Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/takatsuka/GeodesicDome/issues).

## Development set-up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"         # editable install with test and lint tools
pytest                          # runs the test suite
ruff check . && ruff format .   # lint and format
```

Please add or update tests in `tests/`, add a line to `CHANGELOG.md` under *Unreleased*, and keep
the `SPDX-License-Identifier` and copyright header at the top of every source file.

## Licence of contributions (please read)

GeodesicDome is published under the GNU Affero GPL v3 or later (AGPL-3.0-or-later), with an additional attribution term (see
`LICENSE` and `NOTICE`). The copyright holder, Masahiro Takatsuka, also offers the software under other
licences, including commercial ones. To keep that possible, contributions are accepted only on the
following terms.

By submitting a contribution (a pull request, patch, or other material) you confirm that:

1. **You have the right to submit it.** The contribution is your original work, or you otherwise have the
   right to submit it under these terms. If your employer has rights to your work, you have its permission.
2. **You keep your copyright,** and license the contribution to the public under the GNU AGPL v3 or later,
   like the rest of the project.
3. **You also grant a broader licence to Masahiro Takatsuka.** This is a perpetual, worldwide,
   non-exclusive, royalty-free, irrevocable licence, including under any patent claims you can license that
   the contribution necessarily infringes. It covers using, reproducing, modifying, distributing and
   sublicensing the contribution, and licensing it under any other terms, including proprietary or
   commercial ones, alone or as part of GeodesicDome or any other work.
4. **You are not owed any payment or support,** and the contribution is provided "as is", without
   warranty.

Please state in your pull request: *"I agree to the licence terms in CONTRIBUTING.md."* Pull requests
without that statement cannot be merged.
