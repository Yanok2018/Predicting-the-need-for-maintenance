Overview
Keeping the truck fleet in good working order is a crucial task for transport companies. Vehicle malfunctions can lead to significant costs and delays in the delivery of goods. Trucks operating on the Lviv-Kyiv route, sometimes deviating from the route, are exposed to high loads due to long distances, road conditions, and difficult weather conditions. Each breakdown or emergency stop requires considerable financial resources for repairs and time during which the vehicle is idle. In freight transport, unexpected vehicle breakdowns can cause a chain effect, leading to delays in delivery schedules and loss of customer trust.

To address these challenges, this project proposes a system for predicting vehicle maintenance needs based on historical and real-time data. The system uses machine learning algorithms to analyze factors like mileage, load, and operating conditions to predict potential breakdowns, enabling proactive maintenance and reducing unexpected downtimes.

Relevance / Significance
This project is significant as it improves the reliability of transportation, reduces downtime, and cuts maintenance costs. It directly benefits transport companies, drivers, and logistics services by ensuring more stable vehicle operations and timely deliveries. By adopting a data-driven approach to maintenance, the project enhances operational efficiency and adds value to the company's long-term success.

Key Features:
Predictive Maintenance: Forecasts when a vehicle is likely to fail or needs repairs.
Real-Time Data Processing: Monitors vehicle data like mileage and load in real time to make accurate predictions.
Cost Reduction: Minimizes costly emergency repairs by anticipating when maintenance is required.
Improved Fleet Reliability: Ensures that vehicles operate smoothly, reducing downtime and optimizing fleet performance.

Installation
To run this project, you need Python 3.x installed on your machine. Follow these steps to set up the environment and install dependencies:

1. Clone the repository:
git clone Yanok2018/Predicting-the-need-for-maintenance
cd <repository-directory>

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the Flask web application:
flask run

The application will be available at http://127.0.0.1:5000/ by default.

Dependencies
The following Python packages are required to run this project:
Flask: Web framework used to build the web application.
pandas: Data analysis library used for handling and processing vehicle data.
scikit-learn: Machine learning library used for building and training prediction models.
numpy: A core package for numerical computations.
requests: Library for making HTTP requests, used to interact with external services if needed.
gunicorn: A WSGI server for running the Flask app in production.

You can install all dependencies by running:
pip install -r requirements.txt

Usage
1. Login
The system requires users to log in to access the vehicle maintenance data. The login page allows users to authenticate by providing their credentials.
2. Dashboard
Once logged in, users can view the dashboard, where they can input vehicle repair data such as vehicle number, repair date, repair type, and cost. The dashboard also displays the vehicle repair history.
3. Predictions
The system uses machine learning to predict potential vehicle failures or maintenance requirements based on the input data. These predictions are displayed on the "Predictions" page.

Contributing
If you want to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. All contributions are welcome!

License
This project is licensed under the MIT License.