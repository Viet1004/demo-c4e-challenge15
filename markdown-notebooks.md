---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Notebooks with MyST Markdown

Jupyter Book also lets you write text-based notebooks using MyST Markdown.
See [the Notebooks with MyST Markdown documentation](https://jupyterbook.org/file-types/myst-notebooks.html) for more detailed instructions.
This page shows off a notebook written in MyST Markdown.

## An example cell

With MyST Markdown, you can define code cells with a directive like so:

```{code-cell}
print(2 + 2)
```

When your book is built, the contents of any `{code-cell}` blocks will be
executed with your default Jupyter kernel, and their outputs will be displayed
in-line with the rest of your content.

```{seealso}
Jupyter Book uses [Jupytext](https://jupytext.readthedocs.io/en/latest/) to convert text-based files to notebooks, and can support [many other text-based notebook files](https://jupyterbook.org/file-types/jupytext.html).
```

## Create a notebook with MyST Markdown

MyST Markdown notebooks are defined by two things:

1. YAML metadata that is needed to understand if / how it should convert text files to notebooks (including information about the kernel needed).
   See the YAML at the top of this page for example.
2. The presence of `{code-cell}` directives, which will be executed with your book.

That's all that is needed to get started!

## Quickly add YAML metadata for MyST Notebooks

If you have a markdown file and you'd like to quickly add YAML metadata to it, so that Jupyter Book will treat it as a MyST Markdown Notebook, run the following command:

```
jupyter-book myst init path/to/markdownfile.md
```


### Max-Stable Process for Spatiotemporal Data

Max-stable processes are a fundamental tool in extreme value theory, particularly for modeling spatial and spatiotemporal extremes. These processes extend the concept of univariate extreme value distributions to higher dimensions, enabling the analysis of extreme events over space and time.

In the context of spatiotemporal data, max-stable processes are used to model the joint behavior of extreme values across multiple locations and time points. They provide a framework to capture dependencies between extremes, which is crucial for understanding phenomena such as heatwaves, heavy rainfall, or other rare events that exhibit spatial and temporal correlations.

#### Mathematical Representation

A max-stable process \( Z(s) \) for \( s \in \mathcal{S} \), where \( \mathcal{S} \) is a spatial domain, can be defined as:

\[
Z(s) = \max_{i=1, \dots, n} \left\{ \xi_i \cdot W_i(s) \right\},
\]

where:
- \( \xi_i \) are independent and identically distributed random variables following a standard Fréchet distribution,
- \( W_i(s) \) are non-negative stochastic processes satisfying certain conditions to ensure max-stability.

The finite-dimensional distributions of a max-stable process can be expressed as:

\[
P(Z(s_1) \leq z_1, \dots, Z(s_k) \leq z_k) = \exp \left( -V(z_1, \dots, z_k) \right),
\]

where \( V(z_1, \dots, z_k) \) is the so-called exponent measure, which captures the dependence structure of the process.

#### Applications

By leveraging max-stable processes, we can analyze the frequency, intensity, and spatial extent of extreme events, offering valuable insights for risk assessment and decision-making in fields such as climate science, hydrology, and energy systems.