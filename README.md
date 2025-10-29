#  Waste Management Analytics & Predictor

## Overview

This is a comprehensive, interactive web application for waste management analytics and recycling rate prediction. Built with Streamlit and powered by advanced machine learning models, it provides insights into urban waste management patterns across major Indian cities.


---

## Project Structure

```
waste_management/
├── Notebooks/ 
│   ├── data_preparation.ipynb 
│   ├── exploratory_data_analysis.ipynb 
│   ├── feature_engineering.ipynb
│   ├──model_selection.ipynb
│   └── model_training.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
│    ├── catboost_tuned_model.cbm
│    └──catboost_tuned_model.pkl
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── requirements.txt
├── static/ 
├── templates/
├── README.md
├── predictions.csv
└── report.pdf
```
---

## Set up

### 1. Install Dependencies

```bash
# Install the enhanced requirements
pip install -r requirements.txt

# Or install individual packages
pip install streamlit pandas numpy plotly folium streamlit-folium
```

### 2. Run the Application

```bash
# Navigate to the src directory
cd src

# Run the Streamlit app
streamlit run app.py
```

### 3. Access the Application

Open your browser and go to: `http://localhost:8501`

---


## Technical Architecture

### **Frontend**
- **Streamlit**: Modern, responsive web interface
- **Custom CSS**: Enhanced styling and animations
- **Interactive Components**: Forms, charts, maps, and filters

### **Backend**
- **Data Loading**: Cached data and model loading
- **Feature Engineering**: Real-time feature computation
- **ML Prediction**: CatBoost model integration
- **Error Handling**: Robust error management

### **Data Processing**
- **Caching**: Optimized data loading with Streamlit cache
- **Filtering**: Dynamic data filtering and subsetting
- **Aggregation**: Statistical computations and summaries

### **Visualization**
- **Plotly**: Interactive charts and graphs
- **Folium**: Geographic mapping capabilities
- **Responsive Design**: Mobile-friendly interface

## Data Sources

The application uses comprehensive waste management data including:

- **Cities**: Mumbai, Delhi, Bengaluru, Kolkata, Chennai
- **Time Period**: 2019-2023
- **Waste Types**: Plastic, Organic, E-Waste, Construction, Hazardous
- **Features**: 19+ engineered features including:
  - Municipal efficiency scores
  - Landfill capacity and utilization
  - Population density metrics
  - Cost analysis
  - Awareness campaign data

## UI/UX Features

### **Modern Design**
- Gradient color schemes
- Card-based layouts
- Smooth animations and transitions
- Responsive grid systems

### **Interactive Elements**
- Real-time form validation
- Dynamic filtering and sorting
- Progress indicators
- Hover effects and tooltips

### **Accessibility**
- High contrast color schemes
- Clear typography
- Intuitive navigation
- Mobile-responsive design

## AI Prediction Features

### **Model Architecture**
- **Algorithm**: CatBoost Regressor
- **Optimization**: Hyperparameter tuning with Optuna
- **Features**: 19+ engineered features
- **Target**: Recycling Rate (%)

### **Input Parameters**
- City/District selection
- Waste type classification
- Disposal method
- Municipal efficiency scores
- Cost and capacity metrics
- Temporal factors (year)

### **Output Insights**
- Predicted recycling rates
- Confidence intervals
- Feature importance analysis
- Optimization recommendations

---


## Deployment

### **Local Development**
```bash
streamlit run app.py --server.port 8501
```

### **Streamlit Cloud**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy automatically

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_enhanced.txt .
RUN pip install -r requirements_enhanced.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501"]
```

## Future Enhancements

### **Planned Features**
- Real-time data integration
- Advanced ML models (Deep Learning)
- Mobile application
- API endpoints
- Automated reporting

### **Technical Improvements**
- Database integration
- User authentication
- Multi-language support
- Advanced caching strategies



## Predictions File

The repository includes a `predictions.csv` file which stores model predictions along with the corresponding input parameters.  

**How it Works:**  
- When you navigate to the **Predictor** page in the app and use the **Predict Recycling Rate** feature, the entered parameters and the predicted recycling rate are **automatically saved** to `predictions.csv`.  
- The file contains both the **user inputs** and the **model’s prediction** for reference or further analysis.  
- This logging functionality is **only active in local execution**. For deployed versions (e.g., Render), predictions are generated but **not saved** to the file due to server storage limitations.  

**File Structure:**  
| City/District | Year | Waste Type | Disposal Method | Municipal Efficiency | Cost of Waste (₹/Ton) | Landfill Capacity (Tons) | Awareness Campaigns | Predicted Recycling Rate (%) |  
|---------------|------|------------|-----------------|----------------------|-----------------------|--------------------------|---------------------|------------------------------|  

**Example:**  
| Agra          | 2024 | Plastic    | Recycling       | 9                    | 3056 | 500000 | 14 | 66.91 | 




## Acknowledgments

- Streamlit team for the excellent framework
- CatBoost developers for the ML library
- Open source community for visualization tools
- Municipal authorities for data collaboration
- **PWSkills** for organizing the hackathon and providing the platform, dataset, and guidance

---
## Author
Aparna Swain
Data Enthusiast









