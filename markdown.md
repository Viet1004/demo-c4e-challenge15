# Visualizing Climate Impacts on European Energy Systems

## Introduction

The European power system faces significant challenges due to climate variability and extreme weather events. These events impact both electricity consumption and generation, particularly from renewable sources (wind, solar, and hydro). As renewable energy becomes a more prominent part of the energy mix, the power system's vulnerability to a changing climate becomes more visible.

Critical challenges include:
- Compound events mixing cold weather (high demand) with wind droughts (low renewable generation).
- Solar droughts (extended periods of low solar generation).
- Extreme heat leading to increased cooling demand while potentially reducing generation capacity.
- Future uncertainty regarding how these events will evolve under different climate change scenarios.
- Lack of accessible tools for stakeholders to analyze and visualize these climate-driven power system risks.

Recognizing these critical challenges, the European Network of Transmission System Operators for Electricity (ENTSO-E) has developed the Pan-European Climate Database (PECD4.2), which contains comprehensive climate and energy variables derived from ERA5 reanalysis and climate projections. However, despite its immense value, the database's technical complexity, specialized format, and vast data volume create significant barriers for non-specialist stakeholders. This initiative addresses this gap by transforming complex PECD4.2 data into intuitive, interactive visualizations that specifically target high-impact compound climate events affecting energy systems. Given the extensive scope of potential climate impacts, our project strategically focuses on developing scenario-based visualizations that enable grid operators, policymakers, energy developers, and the wider public to identify vulnerabilities, quantify risks, and explore adaptation strategies—ultimately supporting more climate-resilient planning for Europe's evolving power system.

## Exploration of the problem

## Proposed solution

The proposed solution is the development of an interactive visual tool (initially as a Jupyter notebook, with a potential future web interface) to analyse climate-related power system challenges in the EU. This tool will enable users to:

- Select a geographical region of interest, such as a specific country or a group of countries within the Pan-European Climate Database (PECD4.2) domain.
- Define a specific event by setting thresholds for one or more climate and/or energy variables available in the PECD4.2 database. For example, a user could define an event as "average temperature below 0°C AND onshore wind capacity factor below 0.1 lasting for at least 3 consecutive days".
- Analyse the defined event by computing relevant statistics based on both historical data (from the ERA5 reanalysis period, approximately 70 years) and future climate projections (from multiple CMIP6 climate models and SSP scenarios). These statistics will include:
 - The frequency of the event in the historical period.
 - How the frequency, duration, and intensity of the event may change in the future under different SSPs and at different time horizons (e.g., near-term, mid-term, long-term).
 - Visualisations of the spatial aspects of the event (e.g., maps showing the regions affected) and the temporal aspects (e.g., time series plots illustrating event occurrences).
 - Potentially other relevant indicators and statistics to be discussed with mentors.

The tool will be designed to be user-friendly, allowing both specialists and non-specialists to explore the data and gain insights into the climate-related risks facing the EU power system. Clear documentation will be a priority to ensure the tool is accessible and its functionality is well understood.

### Tech stack

We intend to build the whole project around Python and necessary packages.

#### Data Wrangling

For handling NetCDF (nc) files, which is the direct return of the [CDS Data Portal](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=download#technology), we will leverage the following Python libraries:

- **xarray**: This library is particularly suitable for working with multi-dimensional labeled arrays and datasets, which is the native structure of NetCDF files used in the Pan European Climate Database (PECD4.2).
- **pandas**: This library will be essential for data analysis and manipulation, particularly with time series data, which is a significant component of both historical and future climate and energy variables. While the direct download from the CDS might be in NetCDF format, aggregated data is also provided in CSV format, which pandas handles efficiently. Pandas DataFrames can also be used to store and process the results of our analysis, as indicated in the spatial aggregation procedure.
- **numpy**: This foundational library will provide support for numerical computations and array manipulation, underpinning the operations performed by both xarray and pandas.

#### Data Visualization

To enhance the visualizability and interactiveness of our analysis, we aim to use the following Python libraries to better deal with geospatial data and time series:

- **Matplotlib**: This library will serve as a fundamental tool for generating static, interactive, and animated plots. It will be crucial for creating basic visualisations such as time series plots of event occurrences and histograms of event statistics, as envisioned in the project objectives.
- **Plotly**: This library will be considered for creating interactive web-based visualisations, especially if we progress towards developing a web interface as a potential future enhancement.
- **Cartopy**: To better deal with geospatial data, Cartopy will be utilized for creating maps to visualise the spatial aspects of the defined events.

#### Deployment

For deployment, we anticipate three (complementary, not exclusive) outputs, with priority in the following order:

1. **Suite of notebooks**: The primary and minimum expected outcome of this challenge is a well-documented, user-friendly Jupyter notebook, or suite of notebooks, available on GitHub. This aligns directly with the methodology that mentions the minimum requirement is a Jupyter notebook that allows users to select regions and variables, explore the database, and compute desired statistics. The notebooks will serve as the core deliverable, demonstrating how to access the data and visualise the desired statistics.

2. **Publishable Jupyter Book** [(Demo)](https://viet1004.github.io/demo-c4e-challenge15/notebooks/data_exploration.html): To enhance the accessibility, we aim to create a publishable Jupyter Book. This format allows us to weave together narrative text, code, and outputs (including visualisations) into a cohesive and easily navigable document, effectively serving as comprehensive documentation. In particular, to briefly introduce interface with the data platform [CDS Data Portal](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=download#technology), a web-based application in form of a book can be relevant.

3. **Dataportal with dashboard**: As a potential extension, if time allows, we may explore the development of a basic web portal/graphical interface with a dashboard. This would greatly facilitate the use of the tool by non-specialists, providing a more interactive and user-friendly way to define events, analyse data, and visualise results without requiring direct interaction with the Jupyter Notebook environment. This could involve using web development frameworks and interactive visualisation libraries mentioned earlier. This solution can use Jupyter Book as a documentation for first time user. If time allows, we will integrate chatbot with visual understanding to summarize and explain each use case.

### Demo:

We prepare a short demo in a form of jupyter notebook in the following link: [Demo](https://viet1004.github.io/demo-c4e-challenge15/intro.html)

## Timeline

![Timeline](Online%20Gantt%2020250331.pdf)

### Phase 1: Project Setup (Early May)

The first phase focuses on a deep dive into the PECD4.2 database and what the structures of the data are. Simultaneously, a literature review on the power system is conducted to understand and define what compound events are the most critical to the power system and what can be visualized by the PECD4.2 database.

### Phase 2: Basic Implementation (Mid-May to Mid-June)

After defining the scope of the projects, we proceed to the design of the core functionalities that should be included in the product. For example, in addition to basic statistics such as frequency of events in the past and future projection, we can analyze events in the framework of extreme value theory and copulas theory. Upon defining core functionality, we proceed to implement the data pipeline. Initially, one can programmatically acquire the data from the [CDS Data Portal](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=overview) by formulating specific client requests. Subsequently, the data processing phase, incorporating the aforementioned statistical tools, will be conducted.

### Phase 3: Visualization Module (Mid-June to Mid-July)

In this section, we will define use cases (scenarios) which are of interest to various stakeholders in the energy system:

- **Grid Stability Under Extreme Weather Events**: Analysis of how heat waves and cold snaps propagate across regions, creating simultaneous demand spikes that stress transmission infrastructure. Grid operators can identify vulnerable network segments and optimize reserve capacity deployment. Given the energy grids in Europe are tangled, this could be coupled with data from ENTSO-E platform to make it more relevant.
- **Renewable Generation Correlation Risk**: Assessment of joint probability of wind droughts and low solar periods across geographic locations. Energy investors can visualize spatiotemporal dependencies to diversify generation assets and determine optimal storage requirements. Based on different climate scenarios, we can estimate if the investment is worth.

#### (*) Alongside with phase 2 and 3

We will prepare the documentation in the form of a jupyter book, which contains functionality (API reference) and examples/use cases (visualization). By the end of phase 3, we expect to deliver the core functionalities and visualization by mid-term.

### Phase 4: Web-based application (Mid July to late August)

The data portal should allow users to retrieve data and visualize data based on the functionality in Phase 2 and 3. If time allows, the final task involves implementing an AI assistant. By incorporating visual understanding of Vision Language models, this can make first-time, non-expert users familiar with the data format. This tool can process output from statistical functionality in phase 2, combined with its visual understanding to give an overview of the data. This can be done through AI providers or a local model.

## Team members

- Name: NGUYEN Quoc Viet  
 Nationality: Vietnamese  
 Country of residence: Luxembourg  
 Gmail: nguyenquocviet1004@gmail.com  
 Github: https://github.com/Viet1004

- Name: NGO Duc Thinh  
 Nationality: Vietnamese  
 Country of residence: France  
 Gmail: duthngo@gmail.com  
 Github: https://github.com/thinhngo-x

- Name: PHAM Vu Hoang Anh  
 Nationality: Vietnamese  
 Country of residence: France  
 Gmail: hoanganhphamvu9698@gmail.com  
 Github: https://github.com/HoangAnhP

- Name: DAO Nhat Minh  
 Nationality: Vietnamese  
 Country of residence: France  
 Gmail: nhatminh.hp96@gmail.com  
 Github: https://github.com/nhatminh-96

- Name: VU Thi Hai Yen  
 Nationality: Vietnamese  
 Country of residence: France  
 Gmail: haiyen96.hp@gmail.com  
 Github: https://github.com/haiyenvu96