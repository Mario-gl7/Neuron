# Neuron
- The project is divided in three folders: Identifiacion, Segmentation, Diagnostic and Clinical tests:
  
- Identification folder is divided into:
    - Binary (for binary codes):
        - Trans_Long (code that classifies images either Transversal or Longitudinal):
            - Metrics. This folder includes all the metrics calculated. The relevant metrics are from model 8
        - BGF (code that classifies wether images are Bad, Good or Fair):
            - Filters:
                - Trans. The relevant model is 7th.
                - Long. 
            - No filters:
                - Trans. The relevant model is 37th.
                - Long. The relevant model is 23rd.
    - Cathegorical (for more cathegories):
        - Trans. Relevant model is 23rd.
        - Long. Relevant model is 6th.

- Segmentation (_post codes are already trained) folder is divided into:
    - Trans. Relevat model & sgementations is 7th.
    - Long. Relevant model & segmentation is 23rd.

- Diagnostic (post_ codes are already trained) folder is divided into:
    - Trans. Relevant model is 5th.
    - Long. Relevant model is 7th.
  
- Clinical tests. This folder uses the relevant codes of the previous folders and shows a clinical interface, codes are divided in filters and non filters. The following folders contain the result images on each phase of the project:
    - Identification
    - Segmentation
    - Diagnostic

- Data folder contains a json_codes folder where the codes to extract the json from the diagnostic excel are.
