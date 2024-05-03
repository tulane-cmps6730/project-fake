# CMPS 6730 Fake Data Detection

- There have been many fake data detectors out there, yet most of them failed to recongize that there exists bias in training data, that we can expect all news report written in the West has some bias towards the so called "US Adversaries". The goal of this project is to dipit such bias and try to provide a solution for it. 

- The orignal goal was to provide a better dataset, yet it seems that the better way to do this is to better the data preprocessing pipeline to remove all references to specific countries and thereby remove the influence of those countries names. 

- Yet we can still see that now the term "country" becomes very positive by the Lime report that the data is actually bias towards all references of countries, furthermore we can still see terms lying around such as the term "Chinese" or "mainland" as a positive label, that it would pick up on other non-direct references towards countries.  

- Therefore for future work, I purpose a ML word representation model that which would take new article data and try to learn other words that also represent countries in the context of news reporting, and so that we have a better way to remove influence of countries in the data. 
