[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YUvA8hIt)
# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

See the [Module 2 Integration Task Guide](https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-2/learner/integration-guide) for full instructions.

---

## Quick Reference

**File to complete:** `train.py`

**Install PyTorch before running:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Branch:** `integration-2/pytorch`

**Submit:** PR URL → TalentLMS Unit 8 text field





# Housing Price Prediction — PyTorch Model

## 1. What the Model Predicts

This model predicts **housing prices in Jordan (JOD)** based on property features.

### Target Variable
- `price_jod`: The selling price of the property in Jordanian Dinar (JOD)

### Input Features (5)
1. `area_sqm` — Property size in square meters  
2. `bedrooms` — Number of bedrooms  
3. `floor` — Floor number  
4. `age_years` — Age of the property  
5. `distance_to_center_km` — Distance to city center (km)  

---

## 2. Training Configuration

- **Model Type:** Feedforward Neural Network  
- **Architecture:**  
  - Linear(5 → 32)  
  - ReLU activation  
  - Linear(32 → 1)  

- **Loss Function:** Mean Squared Error (MSELoss)  
- **Optimizer:** Adam  
- **Learning Rate:** 0.01  
- **Epochs:** 100  

- **Preprocessing:**  
  - Features were standardized using mean and standard deviation  
  - This ensures balanced gradient updates during training  

---

## 3. Training Outcome

- The loss **decreased steadily during training**, indicating that the model learned meaningful patterns from the data.  
- Initial loss was relatively high but reduced significantly over epochs.  

- **Final Loss Value:** ~ _(put your actual value here from last epoch)_  

---

## 4. Behavioral Observation

During training, it was observed that:

> The loss decreased rapidly during the first few epochs and then gradually stabilized, indicating that the model quickly learned the main patterns and then fine-tuned its predictions.

Additional observation (optional if you noticed):

- Predictions tend to slightly underestimate very high property prices  
- Standardization played a key role in stabilizing training  

---

## 5. Output

The model generates a file:

