# Contributing to Renku

Want to contribute to this template project? Thanks!
There are many ways to help, and we very much
appreciate your efforts.

The sections below provide guidelines for various types of contributions.

# Bug Reports and Feature Requests

Bugs and feature requests should be reported on `s2s-ai-challenge` gitlab [issue tracker](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge/-/issues/new?issuable_template=bug)


# Merge Requests

Checklist:

- MRs should include a short, descriptive title.
- Small improvements need not reference an issue, but PRs that introduce larger changes or add new functionality should refer to an issue.
- Structure your commits in meaningful units, each with an understandable purpose and coherent commit message. For example, if your proposed changes contain a refactoring and a new feature, make two PRs.

## Steps

### 1. Fork this `s2s-ai-challenge-template` repository [https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template).

### 2. Clone your fork locally using `git <https://git-scm.com/>`_, connect your repository
   to the upstream (main project), and create a branch:

```bash
git clone https://renkulab.io/gitlab/$YOURNAME/s2s-ai-challenge-template.git
cd s2s-ai-challenge-template
git remote add upstream https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template.git
```

### 3. To fix a bug or add feature create your own branch off "master":

```bash
git checkout -b your-bugfix-feature-branch-name master
```

If you need some help with Git, follow this quick start
`guide <https://git.wiki.kernel.org/index.php/QuickStart>`_.

### 4. Install dependencies into a new conda environment::

```bash
conda env create -f environment.yml  # rename name: "base" to name: "s2s-ai"
conda activate "s2s-ai"
```

Now you have an environment called ``s2s-ai`` that you can work in.
You’ll need to make sure to activate that environment next time you want
to use it after closing the terminal or your system.

### 5. Break your edits up into reasonably sized commits::

```bash
git commit -a -m "<commit message>"
git push -u
```

### 6. Create a new changelog entry in ``CHANGELOG.md``:

The entry should be entered as: tbd

<description> (``:pr:`#<pull request number>```) ```<author's names>`_``

where ``<description>`` is the description of the PR related to the change and
``<pull request number>`` is the pull request number and ``<author's names>`` are your first
and last names.

Add yourself to list of authors at the end of ``CHANGELOG.md`` file if not there yet, in
alphabetical order.

### 7. Open your MR on renkulab.io/gitlab under [https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/merge_requests](https://renkulab.io/gitlab/aaron.spring/s2s-ai-challenge-template/-/merge_requests).

- Document your MR
- List closing issues
- Add references
- Show improvement
   
### 8. Your MR will be review and eventually merged by a maintainer.

