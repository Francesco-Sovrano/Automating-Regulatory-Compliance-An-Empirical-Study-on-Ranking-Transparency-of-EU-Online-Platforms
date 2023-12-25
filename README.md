# An Empirical Study on Compliance with Ranking Transparency in the Software Documentation of EU Online Platforms

Welcome to the replication package for the ICSE-SEIS 2024 paper titled ["An Empirical Study on Compliance with Ranking Transparency in the Software Documentation of EU Online Platforms."](http://arxiv.org/abs/2312.14794)

## Abstract

Compliance with the EU's Platform-to-Business (P2B) Regulation is challenging for online platforms, and the assessment of their compliance is difficult for public authorities. This is partly due to the lack of automated tools for assessing the information platforms provide in their terms and conditions (i.e., software documentation), in relation to ranking transparency. That gap also creates uncertainty regarding the usefulness of such documentation for end-users. Our study tackles this issue in two ways. First, we empirically evaluate the compliance of six major platforms, revealing substantial differences in their documentation. Second, we introduce and test automated compliance assessment tools based on ChatGPT and information retrieval technology. These tools are evaluated against human judgments, showing promising results as reliable proxies for compliance assessments. Our findings could help enhance regulatory compliance and align with the United Nations Sustainable Development Goal 10.3, which seeks to reduce inequality, including business disparities on these platforms.

## Repository Contents

This repository comprises various tools, scripts, and data sets essential for replicating the findings of our ICSE-SEIS 2024 paper. Here's a detailed breakdown:

- **[setup_virtualenv.sh](setup_virtualenv.sh)**: A script designed to establish a virtual environment for the project, ensuring isolation and specific dependency versions.
  
- **[run_automated_assessments.sh](run_automated_assessments.sh)**: A shell script crafted to execute the automated assessments elucidated in the research paper.

- **[prolific_survey](prolific_survey)**: This folder contains the data resulting from the large-scale manual assessment on Prolific involving 134 participants. For more details, read Section 6.2 of the paper.

- **[expert_vs_gpt_vs_doxpy](expert_vs_gpt_vs_doxpy)**: This folder contains the code of DoXpert and the ChatGPT-based assessment tool. It also contains the data assessment results produced by the three experts and the the software documentation object of this study. For more details, read Sections 6.1 and 6.3 of the paper.
    - [code/data_analysis](expert_vs_gpt_vs_doxpy/code/data_analysis/experts_vs_tools): Contains scripts dedicated to the analysis of the research data.
    - [code/doxpert](expert_vs_gpt_vs_doxpy/code/doxpert): Houses the source code of the DoXpert tool.
    - [code/gpt_based_approach](expert_vs_gpt_vs_doxpy/code/gpt_based_approach): Directory with scripts that implement the baseline tool leveraging ChatGPT. Inside the [marketplaces](expert_vs_gpt_vs_doxpy/code/gpt_based_approach/marketplaces) and [search_engines](expert_vs_gpt_vs_doxpy/code/gpt_based_approach/search_engines) directories, you will find some pkl files containing the cached outputs of the queries to ChatGPT (v4 and v3.5). If you want to regenerate those outputs, cancel the pkl files and run the assessments.
    - [code/packages](expert_vs_gpt_vs_doxpy/code/packages): Includes custom Python packages used throughout the project which are forked versions of:
        - [DoXpy](https://github.com/Francesco-Sovrano/DoXpy)
        - [DiscoLQA](https://github.com/Francesco-Sovrano/DiscoLQA) 
    - [data/assessment_results](expert_vs_gpt_vs_doxpy/data/assessment_results): Contains the outcomes of the automated and experts evaluations of the technical documentation.
    - [data/platform_docs](expert_vs_gpt_vs_doxpy/data/platform_docs): This directory houses the software documentation data from three major online intermediation services (Amazon, Tripadvisor, and Booking) and three online search engines (Google, Bing, and Yahoo). Our selection was driven by representativeness and audience profile. For details on the number of links and average word count per document, refer to the table below.
    - [data/checklist](expert_vs_gpt_vs_doxpy/data/checklist): Features the checklist instrumental in evaluating the compliance of platform documentation with the P2B Regulation.

```
    Platform       | No. of Links | Avg. Words/Doc 
    ---------------|--------------|---------------
    Amazon         | 5            | 434.4
    Bing           | 16           | 964.06
    Booking        | 7            | 579.42
    Google         | 52           | 1679.5
    Tripadvisor    | 10           | 1653.9
    Yahoo          | 3            | 174
```

## System Specifications

This repository is tested and recommended on:

- OS: Linux (Debian 5.10.179 or newer) and macOS (13.2.1 Ventura or newer)
- Python version: 3.7 or newer

## Forked Repositories

This package uses forked versions of two repositories. The original repositories can be found at:
- [DoXpy Repository](https://github.com/Francesco-Sovrano/DoXpy)
- [DiscoLQA Repository](https://github.com/Francesco-Sovrano/DiscoLQA)

## Environment Setup

In order to run the automated assessments, you need to install a proper environment. To do so, we offer two solutions. 

### Docker-based Solution
The first solution relies on Docker.
Use the following command to download the image from Docker Hub:
```bash
docker pull francescosovrano/dox4p2bcompliance
```

After the download is complete, you can verify that the image is downloaded by using the command `docker images`, which will list all the Docker images available on your system.

Once the download is complete, run the following commands to create and start a new container:
```bash
docker run -it francescosovrano/dox4p2bcompliance bash
```

### Shell Script Solution
Instead, if you don't want to use Docker, in order to create the necessary virtual environment and install the necessary dependencies, run the following script:

```bash
./setup_virtualenv.sh
```

## Installation of OpenAI Keys

To use this package, you must set up two environment variables: `OPENAI_ORGANIZATION` and `OPENAI_API_KEY`. These variables represent your OpenAI organization identifier and your API key respectively.

On UNIX-like Operating Systems (Linux, MacOS):
1. Open your terminal.
2. To set the `OPENAI_ORGANIZATION` variable, run:
   ```bash
   export OPENAI_ORGANIZATION='your_organization_id'
   ```
3. To set the `OPENAI_API_KEY` variable, run:
   ```bash
   export OPENAI_API_KEY='your_api_key'
   ```
4. These commands will set the environment variables for your current session. If you want to make them permanent, you can add the above lines to your shell profile (`~/.bashrc`, `~/.bash_profile`, `~/.zshrc`, etc.)

To ensure you've set up the environment variables correctly:

1. In your terminal or command prompt, run:
   ```bash
   echo $OPENAI_ORGANIZATION
   ```
   This should display your organization ID.
   
2. Similarly, verify the API key:
   ```bash
   echo $OPENAI_API_KEY
   ```

Ensure that both values match what you've set.

## Run the Automated Assessments

After setting up the environment, you can run the automated assessments using:

```bash
./run_automated_assessments.sh
```

The GPT-based assessments, specifically from line 9 to line 17 of the [run_automated_assessments.sh](run_automated_assessments.sh) script, will be the quickest due to the pre-cached output of GPT. On the other hand, DoXpert's assessments may take more time, depending on the number of CPUs and GPUs at your disposal. While a GPU is recommended, it is not mandatory.

We are currently using ChatGPT version 0613, which OpenAI may soon discontinue. If you wish to use different versions, you can easily modify the Python scripts found in the [marketplaces](expert_vs_gpt_vs_doxpy/code/gpt_based_approach/marketplaces) and [search_engines](expert_vs_gpt_vs_doxpy/code/gpt_based_approach/search_engines) folders. Additionally, you should update the GPT model referenced in the `instruct_model` function within the [model_manager.py](expert_vs_gpt_vs_doxpy/code/packages/doxpy/doxpy/models/model_manager.py) script.

## Conclusion

We hope this repository serves as a valuable resource for researchers and practitioners aiming to understand and automate the assessment of regulatory compliance, particularly concerning the EU's P2B Regulation. Feedback and contributions are always welcome!

## Support

For any problem or question, please contact me at `cesco.sovrano@gmail.com`