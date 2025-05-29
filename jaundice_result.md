# AI Advancements in Neonatal Jaundice Diagnosis: A Comprehensive Report

## Executive Summary

This report presents a comprehensive overview of the latest advancements in the diagnosis of neonatal jaundice using artificial intelligence (AI). It focuses on non-invasive methods and transcutaneous bilirubinometry, considering various AI techniques to achieve good accuracy in a timely manner across all clinical settings. Recent studies demonstrate the potential of smartphone-based applications and deep learning models in accurately predicting bilirubin levels and classifying jaundice status, offering promising alternatives to traditional methods, especially in resource-limited settings. However, ethical considerations, including algorithmic bias and data privacy, and regulatory compliance are critical factors that must be addressed to ensure equitable and responsible deployment of AI solutions.

---

## 1. Introduction to Neonatal Jaundice and the Need for AI

Neonatal jaundice, characterized by elevated bilirubin levels in newborns, is a common condition requiring timely and accurate diagnosis to prevent potential neurodevelopmental sequelae. Traditional diagnostic methods often involve invasive blood tests, which can be distressing for infants and resource-intensive. The application of AI offers the potential for non-invasive, rapid, and accurate diagnosis, enhancing the quality of care and reducing the burden on healthcare systems. This report examines the recent progress in AI-driven diagnostic tools for neonatal jaundice, covering various techniques, target populations, and clinical settings.

---

## 2. AI-Based Diagnostic Methods for Neonatal Jaundice

### 2.1 Smartphone-Based Applications

Several studies have explored the use of smartphone-based applications for neonatal jaundice screening. These applications typically use the smartphone camera to capture images of the infant's skin, analyzing color variations to estimate bilirubin levels.

* **BiliSG (Singapore):** A study published in *JAMA Network Open* in December 2024 (PMID: 39661385) detailed the development and validation of the BiliSG app, which utilizes machine learning to screen for neonatal jaundice in multiethnic Asian neonates in Singapore. The app achieved a Pearson r of 0.84, 100% sensitivity, and 70% specificity against total serum bilirubin (TSB) measurements, showcasing its potential as a screening tool.

* **Smartphone-Based ML Application:** Another study in *JAMA Network Open* in December 2024 examined a smartphone-based ML application on 546 neonates. The ML model, which incorporated yellowness indicators from the forehead, sternum, and abdomen, underwent internal-external validation against total serum bilirubin (TSB). Pearson r was 0.84, sensitivity was 100%, specificity was 70%, and area under the receiver operating characteristic curve was 0.89.

* **Picterus Jaundice Pro:** An iterative approach to developing a smartphone-based system (Picterus Jaundice Pro), as described in *JMIR Pediatrics and Parenting* in February 2023 (doi:10.2196/40463), resulted in a CE-certified medical device with a validation study showing a correlation of r=0.84 with TSB, 94% sensitivity, and 71% specificity for detecting severe jaundice (TSB >250 µmol/L) in newborns aged 1-15 days.

These smartphone applications provide a non-invasive and accessible method for jaundice screening, particularly valuable in resource-limited settings where access to traditional diagnostic tools may be limited.

### 2.2 Deep Learning Models

Deep learning models, particularly Convolutional Neural Networks (CNNs), have shown promising results in analyzing infant images to predict bilirubin levels and classify jaundice status.

* **1D Convolutional Neural Network (1DCNN):** A study published in *Scientific Reports* on April 4, 2025 (volume 15, Article number: 11571), described an AI-based method using 1D Convolutional Neural Networks (1DCNN) to predict bilirubin levels from infant images. The model achieved an RMSE of 1.13 and an R-squared score of 0.91 when integrating RGB and HSV color spaces, and a 96.87% accuracy in classifying jaundice status. The study used a dataset of 2,235 images from 745 infants sourced from the NeoJaundice dataset at Xuzhou Central Hospital, employing a 1DCNN model trained on Nvidia A100 GPUs via Google Colab, and achieved the best classification results when categorizing bilirubin levels (0-5, 5-10, >10 mg/dL) using a tenfold cross-validation strategy. This same study was found in multiple learnings.

* **Graph Convolutional Network (GCN):** A *Quantitative Imaging in Medicine and Surgery* study from November 8, 2024, using MRI, found that a deep learning system with a graph attention mechanism (GCN) achieved an AUC of 0.86 and ACC of 0.81 for predicting neonatal hyperbilirubinemia when bilirubin levels exceeded 400 µmol/L, significantly outperforming other models.

### 2.3 Other Machine Learning Techniques

Other machine learning algorithms, such as Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), Random Forests, and XGBoost, have also been investigated for jaundice detection.

* **XGBoost:** A February 2024 study in *BioMedInformatics* using 767 infant images found that an XGBoost model achieved 99.63% accuracy in real-time jaundice detection, outperforming SVM (96.22%), k-NN (98.25%), and Random Forest (98.99%), and was successfully implemented in a user-friendly MATLAB application linked to a USB webcam.

* **Nu-Support Vector Classification (NuSVC):** A study published in *Cureus* on June 9, 2024, utilized various Machine Learning (ML) algorithms to classify neonatal bilirubin levels, with the Nu-Support Vector Classification (NuSVC) model achieving the highest testing accuracy of 62.50%, precision of 61.90%, and recall of 56.52%.

* **LightGBM:** In a 2021 study using data from Suzhou Municipal Central Hospital in China (n=984), ensemble learning with LightGBM achieved an AUC of 0.82 (95% CI 0.785-0.857) in predicting neonatal jaundice (CN220 guideline) by integrating clinical risk factors (CRF) and 36 genetic variants (GV36), a 3% improvement over using CRF alone (AUC 0.792).

---

## 3. Performance and Accuracy

The performance of AI-based diagnostic tools varies depending on the algorithm, dataset, and validation methods used. Some models have demonstrated high accuracy in predicting bilirubin levels and classifying jaundice status.

* **High Accuracy Models:** Studies have reported accuracies exceeding 90% for certain AI models, particularly those using neural networks, in detecting neonatal jaundice (Artif Intell Med. 2025 Apr;162:103088).
* **RMSE and R-squared:** The 1DCNN model achieved an RMSE of 1.13 and an R-squared score of 0.91 in predicting bilirubin levels, indicating a strong correlation between predicted and actual values (Sci Rep 15, 11571 (2025)).
* **AUC and ACC:** The graph attention mechanism (GCN) achieved an AUC of 0.86 and ACC of 0.81 for predicting neonatal hyperbilirubinemia when bilirubin levels exceeded 400 µmol/L (Quantitative Imaging in Medicine and Surgery study from November 8, 2024).

---

## 4. Clinical Settings and Target Populations

AI-based diagnostic tools for neonatal jaundice are applicable across various clinical settings, including primary care, emergency departments, neonatal intensive care units (NICUs), and resource-limited settings.

* **Resource-Limited Settings:** Mobile apps leveraging smartphone cameras offer practical solutions for bilirubin level estimation in resource-limited settings (Artif Intell Med. 2025 Apr;162:103088).
* **Multiethnic Populations:** The BiliSG app was specifically developed and validated for multiethnic Asian neonates in Singapore, demonstrating the importance of considering ethnic diversity in AI model development (JAMA Netw Open. 2024;7(12):e2450260).

---

## 5. Ethical Considerations and Regulatory Compliance

The deployment of AI in healthcare raises significant ethical considerations, including data privacy, algorithmic bias, transparency, clinical validation, and professional responsibility. It is crucial to address these concerns to ensure responsible and equitable deployment.

### 5.1 Algorithmic Bias

Algorithmic bias can perpetuate discrimination based on race, gender, and socioeconomic status, leading to unfair outcomes. Configr Technologies highlighted on February 29, 2024, that algorithmic bias in AI systems can lead to unfair outcomes, erosion of trust, and significant legal and ethical implications across sectors like healthcare, finance, and criminal justice.

### 5.2 Data Privacy

Protecting patient data privacy is essential, and AI systems must comply with relevant regulations such as HIPAA and GDPR. A Cureus review published on 2024-06-15 (PMCID: PMC11249277) highlights ethical considerations for AI/ML in healthcare, emphasizing privacy, algorithmic bias, transparency, clinical validation, and professional responsibility.

### 5.3 Transparency and Explainability

Transparency and explainability are crucial for building trust in AI-based diagnostic tools. Healthcare professionals need to understand how AI algorithms arrive at their conclusions to ensure appropriate clinical decision-making. A study published in *AJNR Am J Neuroradiol* in November 2023 (44(11):1242-1248) emphasizes the importance of explainability, accountability, and transparency in AI algorithm development and clinical deployment in neuroradiology, advocating for fairness criteria that maximize benefit and minimize harm to patients.

### 5.4 Regulatory Compliance

Manufacturers incorporating AI in medical devices must comply with existing regulations like MDR and IVDR, demonstrating device benefits, safety (repeatability, reliability, performance per MDR Annex I, 17.1 or IVDR Annex I, 16.1), and validation against a precise intended purpose (MDR/IVDR Annex II), while also adhering to the EU AI Act, potentially classifying these devices as 'high-risk' (as of April 14, 2025).

The FDA issued draft guidance on January 6, 2025, providing comprehensive recommendations for AI-enabled medical devices throughout their Total Product Life Cycle, emphasizing transparency, bias mitigation, and proactive planning for device updates through predetermined change control plans; public comment was requested by April 7, 2025.

### 5.5 Relevant Standards

ISO 13485:2016, ISO 14971:2019, IEC 62304:2006+AMD1:2015, IEC 62366-1:2015+AMD1:2020, and IEC 81001-5-1 are considered key base standards for AI-based medical devices, while AAMI TIR 34971:2023 and ISO/IEC 5259-2 and -4 provide AI-specific guidance on risk management and data quality, respectively.

---

## 6. Impact on Neurodevelopmental Outcomes

Neonatal jaundice, if left untreated, can lead to neurodevelopmental sequelae. AI-based diagnostic tools can facilitate early detection and intervention, potentially mitigating these risks.

* **Long-Term Sequelae:** A 2020 study in Taiwan following 66,983 neonates with significant neonatal jaundice (SNJ) from 2000-2003 found a significantly higher cumulative rate of long-term neurodevelopmental sequelae compared to a reference cohort (P < 0.05), with SNJ patients exhibiting a 1.5 to 3 times higher risk.

* **BERA Abnormalities:** A 2020 study published in the *Journal of Clinical Neonatology* found a linear association between serum bilirubin levels and both BERA (Brainstem-Evoked Response Audiometry) changes and neurodevelopmental sequelae; a mean serum bilirubin of 22.58 mg/dL was associated with abnormal BERA (P < 0.001), and patients receiving exchange transfusions had higher odds of neurodevelopmental and BERA abnormalities.

It is important to note that a 2006 *NEJM* study of 140 infants with total serum bilirubin levels of $\geq 25 \text{ mg/dL}$ found that when treated with phototherapy or exchange transfusion, these levels were not associated with adverse neurodevelopmental outcomes, with follow-up data available for 132 infants to at least age 2 and formal evaluations at a mean age of 5.1 years. This highlights the importance of appropriate intervention.

---

## 7. AI Research Areas in Neonatology

A November 2023 *NPJ Digital Medicine* systematic review (PMCID: PMC10682088) of 106 neonatology AI research articles from 1996-2022 identifies primary focus areas as survival analysis, neuroimaging, vital parameter analysis, and retinopathy of prematurity diagnosis.

---

## 8. Disparities and Global Impact

The impact of neonatal jaundice varies across different regions, with higher incidence rates in certain areas. AI-based diagnostic tools can help address these disparities by providing accessible and accurate diagnostic solutions in resource-limited settings.

* **Regional Incidence:** A systematic review and meta-analysis published in the *BMJ Paediatrics Open* in 2017 found that the African region had the highest incidence of severe neonatal jaundice (SNJ) at 667.8 per 10,000 live births, followed by Southeast Asia (251.3) and the Eastern Mediterranean (165.7).

* **Prevalence Across WHO Regions:** A 2023 systematic review and meta-analysis in the *Journal of Clinical Medicine* found that the prevalence of SNJ among all hospital admissions varied across WHO regions, with the African region reporting the highest prevalence (3.34%), followed by the South-East Asian region (2.58%).

* **LMIC Jaundice-Related Deaths:** A 2023 study in the *Journal of Clinical Medicine* identified that in LMICs the percentage of jaundice-related deaths were 13.02% and 7.52% in Eastern Mediterranean and African regions respectively.

---

## 9. NIH Initiatives

The NIH Office of Data Science Strategy (ODSS) announced "Administrative Supplements for Advancing the Ethical Development and Use of AI/ML in Biomedical and Behavioral Sciences" on February 3, 2022 and made twenty-two awards to principal investigators at 33 different institutions across the country to address ethical concerns, biases, identifiability, privacy, impacts on disadvantaged groups, health disparities, and adverse social consequences in AI/ML research.

---

## 10. Conclusion

AI-based diagnostic tools hold great promise for improving the diagnosis and management of neonatal jaundice. Smartphone-based applications and deep learning models have demonstrated high accuracy in predicting bilirubin levels and classifying jaundice status, offering non-invasive and accessible alternatives to traditional methods. However, ethical considerations, regulatory compliance, and data quality must be carefully addressed to ensure responsible and equitable deployment. Future research should focus on refining AI algorithms, validating their performance across diverse populations, and integrating them into clinical workflows to enhance the quality of care for neonates.

---

## 11. Future Directions

* **Longitudinal Studies:** Conducting longitudinal studies to assess the long-term impact of AI-based diagnostic tools on neurodevelopmental outcomes.
* **Integration with EHR Systems:** Integrating AI systems with electronic health record (EHR) systems to facilitate seamless data exchange and clinical decision support.
* **Community Engagement:** Engage community stakeholders to address concerns and promote trust in AI-based healthcare solutions.
* **Focus on Health Equity:** A Preventing Chronic Disease commentary issued on 2024-08-22 emphasizes the importance of health equity and ethical AI use in public health and medicine, recommending community engagement, inclusive data practices, and transparent algorithms to ensure AI benefits all populations equitably.
* **Trust Factors:** A BMC Medical Informatics and Decision Making study from 2023-01-13 identifies data quality, algorithmic bias, opacity, safety/security, and responsibility attribution as key factors affecting trust in medical AI, emphasizing the need for ethical governance through data management, transparency, and risk assessment.

---

## Sources

* https://pmc.ncbi.nlm.nih.gov/articles/PMC9562034/
* https://configr.medium.com/algorithmic-bias-ai-ethics-a188f54efc96
* https://pmc.ncbi.nlm.nih.gov/articles/PMC11652016/
* https://pubmed.ncbi.nlm.nih.gov/39988547
* https://www.nature.com/articles/s41746-023-00941-5
* https://journals.lww.com/jocn/fulltext/2020/09020/neurodevelopmental_outcome_at_6_months_of_age_in.7.aspx
* https://pmc.ncbi.nlm.nih.gov/articles/PMC10682088/
* https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2022.880034/full
* https://www.researchgate.net/publication/389134321_Artificial_Intelligence_non-invasive_methods_for_neonatal_jaundice_detection_A_review
* https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2827752
* https://www.researchgate.net/publication/389705420_Ethical_Considerations_in_AI_Healthcare_Solutions
* https://pmc.ncbi.nlm.nih.gov/articles/PMC10631523/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC8638201/
* https://www.mdpi.com/2673-7426/4/1/34
* https://bmjpaedsopen.bmj.com/content/1/1/e000105
* https://academicstrive.com/JNRPC/JNRPC180053.pdf
* https://joirem.com/wp-content/uploads/journal/published_paper/volume-05/issue-4/J_OUv6cZ2j.pdf
* https://pmc.ncbi.nlm.nih.gov/articles/PMC10253859/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC11249277/
* https://blog.johner-institute.com/regulatory-affairs/regulatory-requirements-for-medical-devices-with-machine-learning/
* https://www.researchgate.net/publication/356698772_Ensemble_learning_for_the_early_prediction_of_neonatal_jaundice_with_genetic_features
* https://publications.aap.org/pediatrics/article/114/1/297/64771/Management-of-Hyperbilirubinemia-in-the-Newborn
* https://www.modernpathology.org/article/S0893-3952(24)00266-7/fulltext
* https://pmc.ncbi.nlm.nih.gov/articles/PMC11233198/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC9840286/
* https://www.nejm.org/doi/full/10.1056/NEJMoa054244
* https://pmc.ncbi.nlm.nih.gov/articles/PMC7347619/
* https://www.hardianhealth.com/insights/regulatory-ai-medical-device-standards
* https://www.hopkinsmedicine.org/-/media/files/allchildrens/clinical-pathways/jhach-neonatal-hyperibilirubinemia-clinical-pathway-rev-12-june-2023v2.pdf
* https://www.researchgate.net/publication/378270921_Ethical_Considerations_in_AI_Addressing_Bias_and_Fairness_in_Machine_Learning_Models
* https://pubmed.ncbi.nlm.nih.gov/39661385
* https://www.fda.gov/media/184856/download
* https://www.cdc.gov/pcd/issues/2024/24_0245.htm
* https://www.researchgate.net/publication/374106948_A_Mortality_Prediction_System_for_Neonatal_Jaundice_Using_Machine_Learning_Techniques
* https://www.researchgate.net/publication/390494376_Artificial_intelligence-based_non-invasive_bilirubin_prediction_for_neonatal_jaundice_using_1D_convolutional_neural_network
* https://pediatrics.jmir.org/2023/1/e40463/
* https://pmc.ncbi.nlm.nih.gov/articles/PMC11635536/
* https://www.sciencedirect.com/science/article/abs/pii/S0146000520301361
* https://www.binasss.sa.cr/abr25/12.pdf
* https://www.sciencedirect.com/science/article/pii/S2162098923001020
* https://datascience.nih.gov/artificial-intelligence/initiatives/ethics-bias-and-transparency-for-people-and-machines
* https://www.fda.gov/news-events/press-announcements/fda-issues-comprehensive-draft-guidance-developers-artificial-intelligence-enabled-medical-devices
* https://pubmed.ncbi.nlm.nih.gov/39988547/
* https://www.bsigroup.com/globalassets/meddev/localfiles/en-gb/documents/bsi-md-de-qa-medical-webinar-samd-uk-en.pdf
* https://www.nature.com/articles/s41598-025-96100-9
* https://pmc.ncbi.nlm.nih.gov/articles/PMC9402995/
* https://dl.acm.org/doi/abs/10.1016/j.artmed.2025.103088