# Shades of Red and Blue

An comprehensive, unbiased political quiz that uses actual data to analyze American politics and place people on a spectrum. 

## Intro

A "political spectrum" serves as a visual representation of diverse political ideologies, using a grid where each point corresponds to a unique viewpoint. This grid consists of different "axes," representing midpoints between opposing extremes. Positions on these axes function as coordinates, uniquely defining a political stance. Political entities such as groups, parties, candidates, and everyday citizens are characterized based on their positions on these axes. While a simple left-right spectrum is a common conceptualization globally, alternatives, such as the Nolan Chart, introduce an orthogonal axis representing attitudes toward government control, ranging from libertarianism to totalitarianism[1].

Websites like <https://www.politicalcompass.org> aim to provide a straightforward way for individuals to determine their placement on this spectrum. This analysis of political ideology has gained traction in popular internet culture, as evident in places like <https://www.reddit.com/r/PoliticalCompassMemes/>. However, the subjective nature of defining axes and political concepts like "left," "right," "authoritarian," or "libertarian" leads to varied interpretations. This project addresses this issue by seeking a more objective method of constructing a political spectrum.

Using techniques from machine learning along with data from political surveys, we can construct a version of this chart from data instead of relying on intuition. This is done by taking

[1]: https://polquiz.com/

## Methodology

Data is taken from the American National Election Survey's 2020 survey, which consists of a comprehensive political questionnaire of subjects before and after the 2020 US Presidential election. Features relevant to political views, which included direct questions on political policy as well as demographic information. Ideological questions were then put through a PCA model using weights given by the dataset.

## Quick Start

## Usage


