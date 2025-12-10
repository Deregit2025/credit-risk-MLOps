## Credit Scoring Business Understanding

Credit scoring is the process of quantitatively assessing the creditworthiness of a potential borrower. In this project, we focus on predicting the likelihood of default for customers of a Buy-Now-Pay-Later (BNPL) service using alternative transaction data provided by an eCommerce platform.

---

### 1. Basel II Accord’s Influence

The Basel II Capital Accord emphasizes the measurement, management, and disclosure of credit risk for banks. Key implications for this project include:

- **Interpretability:** Models must be understandable and justifiable to regulators and stakeholders.  
- **Documentation:** Assumptions, data sources, and modeling logic must be well-documented to meet compliance requirements.  
- **Risk Measurement:** Accurate estimation of default probabilities and risk exposure is required to inform lending decisions.  

Thus, our credit scoring model is designed to be **both predictive and interpretable**, balancing performance with regulatory requirements.

---

### 2. Necessity of a Proxy Variable

The dataset does not include a direct label indicating whether a customer defaulted. To overcome this, we create a **proxy target variable** representing “high-risk” customers based on behavioral metrics (e.g., Recency, Frequency, and Monetary value).  

**Business considerations:**

- **Why necessary:** Without a proxy, no supervised learning model can be trained.  
- **Potential risks:** Proxy-based predictions may misclassify customers, leading to potential financial loss or missed lending opportunities.  
- **Mitigation:** Carefully engineered RFM-based clustering and validation help reduce errors and increase reliability.

---

### 3. Model Choice Trade-Offs

In a regulated financial environment, there are trade-offs between **simple, interpretable models** and **complex, high-performance models**:

| Model Type | Pros | Cons | Use Case in Credit Risk |
|------------|------|------|------------------------|
| Logistic Regression with WoE | Interpretable, regulator-friendly, easy to document | May underperform on complex patterns | Suitable for baseline or regulated context |
| Gradient Boosting (XGBoost/LightGBM) | High predictive accuracy, captures non-linear interactions | Less interpretable, harder to explain to regulators | Used when performance is critical and interpretability tools are applied (e.g., SHAP) |

Our approach balances **regulatory compliance and predictive performance**, using interpretable features like WoE and RFM metrics while experimenting with both simple and complex models.

---

### Summary

- The project addresses **default risk prediction** for BNPL customers.  
- A **proxy target variable** is necessary due to missing default labels.  
- Model development will carefully balance **interpretability** (for Basel II compliance) and **predictive performance**.
