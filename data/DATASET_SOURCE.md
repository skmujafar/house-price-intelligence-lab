# Dataset source and scope

Dataset: California Housing, provided by scikit-learn.
Official documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
Original source: StatLib / Carnegie Mellon University.
Observation unit: California census block groups, derived from 1990 U.S. Census data.
Size: 20,640 observations; 8 numeric features and a target.
Target: `MedHouseVal` in units of $100,000; `MedHouseVal_USD` is a convenience conversion.

This is real-world historical aggregated data, not current individual-property listings and not Indian housing data. The application downloads it on first run if the CSV is not present. Internet access is required for the initial download.
