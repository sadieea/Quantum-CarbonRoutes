# ⚛️ Quantum CarbonRoute  
### Quantum-Inspired + AI-Driven Carbon-Optimized Route Planning

**Quantum CarbonRoute** is a hybrid **Quantum-inspired optimization** + **AI prediction** system that generates ultra-efficient delivery routes designed to **minimize CO₂ emissions**, fuel cost, and travel distance.


---

## 🌍 Problem We’re Tackling

Delivery fleets waste fuel and produce unnecessary CO₂ because of:

- ❌ Sub-optimal routing  
- ❌ Incorrect traffic or demand prediction  
- ❌ Poor vehicle load balancing  
- ❌ Lack of real-time carbon feedback  

Traditional VRP solvers slow down as problem size grows.

**Quantum CarbonRoute** solves this using a **LD-DAQC inspired quantum model** + **AI-assisted demand estimation**.

---

## ⚡ What Makes Our Approach Special

### 🔮 Quantum-Inspired QUBO Optimization  
We model the Vehicle Routing Problem (VRP) as a **QUBO** and solve it using  
**Qiskit Aer’s quantum simulator**, inspired by **LD-DAQC (Lagrangian Duality-Discretized Adiabatic Quantum Computation)**.

### 🤖 AI-Enhanced Route Inputs  
AI (NumPy/Pandas + forecasting logic) enhances:

- Node clustering  
- Load balancing  
- Emission estimates  

(Currently simple; scalable to LSTM/LightGBM later.)

### 💚 Carbon Footprint Score  
Our CO₂ model converts:  
**Distance → Fuel → CO₂ → Savings %**

### 🗺️ Interactive React UI  
The optimized routes are visualized using **Leaflet**, with:

- Multi-vehicle route coloring  
- Hover-based stop info  
- CO₂ saved, fuel saved, and distance metrics  


---

## 🛠️ How to Run the Project (Correct & Tested)

### 🧠 Run Backend (FastAPI + Qiskit)


```bash
pip install uvicorn fastapi numpy qiskit qiskit-aer pandas python-dotenv

python -m uvicorn main:app --reload --reload-exclude "react-app/*"

```

Backend runs at:

👉 http://localhost:8000

👉 http://localhost:8000/docs
 (Swagger API)
 
---


### 💻 Run Frontend (React UI)

```bash
cd react-app
npm install
npm start
```

Frontend runs at:

👉 http://localhost:3000
 
---


## 🧠 Tech Stack — Accurate & Updated

### **Quantum / Optimization**
- Qiskit Aer simulator  
- QUBO modeling  
- DAQC-inspired adiabatic logic  
- NumPy for Hamiltonian math  
- *(No D-Wave — this project uses Qiskit only)*

### **AI / Data**
- Pandas (data handling)  
- NumPy (matrix transformations)  
- Light predictive logic (demand estimation)

### **Backend**
- FastAPI  
- Uvicorn  
- Python 3.10+

### **Frontend**
- React  
- Leaflet Map  
- Axios  


---

## 🎯 Core Features

### ⚛️ **Quantum-Inspired Route Solver**
- Minimizes distance + CO₂ simultaneously  
- Handles capacity and emission constraints  
- Outputs optimized node ordering  

### 🌡️ **Carbon Engine**
- Converts distance → fuel → CO₂  
- Calculates CO₂ savings vs naive routes  

### 🧠 **AI Assistance**
- Zone grouping  
- Load estimation  
- Simple demand inference  

### 🗺️ **React Visualization**
- Full map-based route display  
- Vehicle route colors  
- Live metrics panel  

---

## 👩‍💻 Developer

**Sadiya Ansari**  
Developer • Quantum Algorithm Architect • AI Engineer  

Designed, developed, and integrated the complete end-to-end system.

---

## 🎥 Demo Video  
[Youtube Video](https://www.youtube.com/watch?v=tH98jIjqg601)

---

## 📜 License  
MIT License  

---

## ⭐ Like the Project?  
Star the repo to support future quantum-AI hybrids ✨


