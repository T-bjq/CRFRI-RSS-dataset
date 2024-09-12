## Overview
This project introduces the Cross-Regional Fire Detection and Rescuer Identification (CRFRI) system, a novel approach leveraging Wireless Sensing Technology. Unlike traditional fire detection methods that depend on sensors and cameras, CRFRI uses changes in Received Signal Strength (RSS) from wireless signals to detect fires and identify individuals in need of rescue, even in complex, non-line-of-sight (NLOS) scenarios. This method offers a cost-effective, practical solution to improve rescue operations in fire-affected areas.

## Key Features
- **Non-Line-of-Sight (NLOS) Detection**: Detects fire and identifies individuals through walls using RSS variations, overcoming limitations of traditional line-of-sight methods.
- **High Accuracy**: Achieves a success rate of over 98.33% in identifying fire incidents and individuals needing rescue.
- **Cost-Effective**: Leverages existing wireless infrastructure, reducing the need for specialized equipment.
- **Rescue Prioritization**: Assesses the presence and location of individuals in fire-affected areas, aiding in effective allocation of rescue resources.

## How It Works
- **Data Collection**: Utilizes Wi-Fi System-on-Chip (SoC) ESP32 to capture RSS data from multiple locations.
- **RSS Analysis**: Monitors changes in RSS due to temperature increases and smoke, using machine learning models to classify fire and non-fire conditions.
- **Feature Extraction**: Incorporates advanced deep learning techniques, including convolutional layers and attention mechanisms, to enhance feature extraction and classification.
- **Rescuer Identification**: Determines the presence of individuals by analyzing RSS patterns, allowing for real-time prioritization of rescue operations.

## Experimental Setup and Results
- **Scenarios**: Conducted in various environments with distances ranging from 1 to 5 meters from the fire source.
- **Performance**: Demonstrated high accuracy in both fire detection and identification of individuals, outperforming traditional methods and several existing models (e.g., ResNet, MobileNet).
- **Metrics**: Evaluated using standard metrics such as accuracy, precision, and recall; proved to be efficient with low computational cost and test time.

## Advantages Over Traditional Methods
- **Improved Coverage**: Extends fire detection to multiple rooms or areas without direct line-of-sight.
- **Lower Costs**: Utilizes existing wireless networks without needing additional equipment.
- **High Efficiency**: Enhances rescue operations by accurately identifying the number and location of individuals.

## System Architecture
- **Wireless Signal Processing**: Explores how wireless signals are affected by fire-induced changes, such as temperature increase and smoke.
- **Machine Learning Models**: Applies multiscale convolutional layers and attention mechanisms for accurate classification.
- **Hardware Configuration**: Deploys ESP32 Wi-Fi SoCs for continuous RSS data collection and real-time analysis.

## Conclusion and Future Work
The CRFRI system provides an innovative, cost-effective solution for fire detection and rescuer identification, particularly in NLOS environments like residential buildings and shopping malls. Future research will focus on optimizing detection algorithms and improving the system's adaptability to various environmental conditions.

## Contact Information
For further information or questions, please contact the authors via the details provided in the full paper.