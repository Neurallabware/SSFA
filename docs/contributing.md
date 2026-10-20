# Maintain and build these docs

Documentation lives next to the implementation. Update the guide and source references whenever a CLI, array layout, default, checkpoint location, or output meaning changes.

## Build locally

Use a separate Python 3.12 environment. From the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -n -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`. The documentation environment installs only documentation tooling. It does not import the scientific package, execute notebooks, download datasets or weights, or start GPU jobs.

The Documentation workflow runs the same strict HTML build for documentation changes. A passing docs build checks documentation structure and references; scientific examples need their own runtime and data validation.

## Review checklist

- Link each operational claim to the implementation or configuration that defines it.
- State array axes, units, sampling rate, normalization, component IDs, and any resampling.
- Distinguish a documented command, a completed run, and an independently validated result.
- Keep unpublished data, credentials, personal information, and machine-specific credentials out of examples.
- Link upstream documentation and describe local changes instead of copying an upstream manual.
- Check a small authorized dataset separately before documenting a command as runtime-tested.

## Read the Docs setup

The repository includes `.readthedocs.yaml`; it does not create a hosted project by itself.

1. Import this repository into the existing Read the Docs account.
2. Use Read the Docs Business for private source repositories or restricted documentation. Keep the existing source and documentation access policy.
3. Select the branch or release to build, verify the HTML build, and enable pull-request previews if required.
4. Record the verified project URL in the repository README. Add the project to the calcium portal only after its hosted URL works.
5. If using a parent project, register this site as a subproject. Subproject versions remain independent.

Creating a new paid subscription or making private material public requires a separate owner decision. See the official [configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html), [subprojects guide](https://docs.readthedocs.com/platform/stable/subprojects.html), and [private hosting guide](https://docs.readthedocs.com/platform/stable/commercial/index.html).
