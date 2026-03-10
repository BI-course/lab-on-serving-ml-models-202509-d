# Windows Quickstart for Beginner + Intermediate Submission

## 1) Run the API (PowerShell)

```powershell
cd C:\Users\benny\lab-on-serving-ml-models-202509-d
.\.venv\Scripts\Activate.ps1
python api.py
```

API base URL while running locally:

```text
http://127.0.0.1:5000
```

## 2) Test Beginner Endpoints

### Decision Tree Classifier

```powershell
$body = @{
    monthly_fee = 60
    customer_age = 30
    support_calls = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions" `
    -Method POST -Body $body -ContentType "application/json"
```

### Decision Tree Regressor

```powershell
$body = @{
    PaymentDate = "2027-11-13"
    CustomerType = "Business"
    BranchSubCounty = "Kilimani"
    ProductCategoryName = "Meat-Based Dishes"
    QuantityOrdered = 8
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions" `
    -Method POST -Body $body -ContentType "application/json"
```

## 3) Test Intermediate Classifier Endpoints

Use this payload for Naive Bayes, SVM, and Random Forest:

```powershell
$onlineShoppers = @{
    Administrative = 0
    Administrative_Duration = 0.0
    Informational = 0
    Informational_Duration = 0.0
    ProductRelated = 2
    ProductRelated_Duration = 64.0
    BounceRates = 0.02
    ExitRates = 0.04
    PageValues = 0.0
    SpecialDay = 0.0
    Month = "May"
    OperatingSystems = 1
    Browser = 1
    Region = 1
    TrafficType = 2
    VisitorType = "Returning_Visitor"
    Weekend = $false
} | ConvertTo-Json
```

### Naive Bayes

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/naive-bayes-classifier/predictions" `
    -Method POST -Body $onlineShoppers -ContentType "application/json"
```

### Support Vector Machine

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/support-vector-classifier/predictions" `
    -Method POST -Body $onlineShoppers -ContentType "application/json"
```

### Random Forest

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/random-forest-classifier/predictions" `
    -Method POST -Body $onlineShoppers -ContentType "application/json"
```

### kNN

```powershell
$knnBody = @{
    Customer_rating = 4
    Cost_of_the_Product = 250
    Prior_purchases = 2
    Discount_offered = 10
    Weight_in_gms = 3000
    Shipping_Mode = "Standard Class"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/models/knn-classifier/predictions" `
    -Method POST -Body $knnBody -ContentType "application/json"
```

## 4) Test Intermediate Recommender Endpoint

```powershell
$recoBody = @{
    items = @("whole milk", "root vegetables")
    top_n = 3
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/v1/recommenders/association-rules/recommendations" `
    -Method POST -Body $recoBody -ContentType "application/json"
```

## 5) Submission Checklist

- Fill `lab_submission_instructions.md`:
  - Team details + each member branch link + what each learned
  - Chosen level = Intermediate
  - Video link (<= 5 minutes)
  - Public Gradio or Streamlit URL
- Push code changes to GitHub
- Deploy one model publicly:
  - `streamlit-sharing-using-streamlit/app.py` or
  - `huggingface-spaces-using-gradio/app.py`
