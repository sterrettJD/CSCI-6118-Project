# An Interactive Educational Resource for the Analysis of Microbial Community Ecology

## Files contained
- Notebooks/ contains Jupyter notebooks that will be used to build the book
- index.html contains the first page for the book
- utils.py contains utilities for interactive examples
- environment.yml contains information for the conda environment (for Binder)
- _toc.yml contains the table of contents for Jupyter book
- _config.yml contains the configuration settings for Jupyter book


## Project Pitch

[Google Slides Link](https://docs.google.com/presentation/d/1sRPDgFCIUhIjcCYBGer84pVo_-rfNCY0zqiiBznzKnU/edit#slide=id.p)

### From Keybase pitch:

Effectively, we would create a Jupyter notebook (or a series of Jupyter notebooks) in Python, 
which contain descriptions of concepts like alpha and beta diversity and PCoA, 
as well as demonstrations of how these are calculated based on simulated data. 
Ideally, we could make a few ways to customize these using point-and-click methods and/or code to see how the outcomes change with differing populations/parameters. 
To make this code accessible to others, I think the easiest route would be to Binder-ize (https://mybinder.org/) the notebooks or 
turn them into a [Jupyter Book](https://jupyterbook.org/intro.html). 
This would allow for others to open the resource and run/modify the code in an online interface with the conda environment already set up for them. 
For an example, feel free to check out Introduction to Applied Bioinformatics ([Jupyter book](https://readiab.org/introduction.html) ; 
[Binder](https://mybinder.org/v2/gh/applied-bioinformatics/iab2/main?urlpath=tree/book/introduction.md)). 
I've seen some folks publishing their code for papers in Binder, so it might be a useful skill to learn for the sake of reproducibility! 

### Interactive plotting?

See Plotly examples
- Jupyter book actually shows [example](https://jupyterbook.org/interactive/interactive.html) info for hosting interactive figures!
- Use FigureWidget? -> see [example](https://plotly.com/python/figurewidget-app/)

### Binderizing/Jupyterbook
- Website for [JupyterBook](https://jupyterbook.org/intro.html)
- [How to make your first JupyterBook](https://jupyterbook.org/start/overview.html)
- Website for [Binder](https://mybinder.org/)
- [Executable Book Project](https://executablebooks.org/en/latest/) - this is what supports Jupyter Book and Binder


### Assignment

For your project pitch you will record a short presentation of slides that outline your project (you can record via Zoom or any recording software of your choice). All students in the group must present equal portions of the talk. Think of this talk as if it were for a conference.

**Duration: 15 minutes**

Submission: Uploaded project pitch video to your Keybase group channel on/before due date

Content (all times are approximate)

**Background/Overview (5 minutes)**

The background should provide a comprehensive scientific overview of the project that is aimed at a general scientific audience. Keep the discipline specific jargon to a minimum and define terms when needed. This overview should frame the problem and indicate why a software development solution is necessary.

**Software Architecture (5 minutes)**

This portion of the presentation should include the high-level design/architecture of the software you intend to develop. This will expand upon the problem framed in the background section and provide added technical context. You should describe the inputs and outputs of the software—i.e. describe the format of the data and any constraints or limitations that would be computationally limiting; you should describe the nature/format of the results your software is intended to produce.

**Technical Implementation (5 minutes)**

This portion should include a more detailed description of how you plan to achieve/implement your high-level design. Here you will want to outline the different data types you’ll be using to store the data, what kinds of error checking you’ll need to perform, what computational bottlenecks you are anticipating, how you will organize your code, how you plan to divide up the different portions of the project among the developers, etc.

**Note**
I would suggest that the project lead (the person who proposed the project) should present the background/overview section and the most experienced coder (or simply the other team member, if it’s a group of two) should present the technical implementation and/or architecture portion. However, it is the job of the project lead to bring their team up to speed on the background and motivation for the project, and ALL members of the group should be contributing to the technical details of the implementation even if only one person presents on it.
