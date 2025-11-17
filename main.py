from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from quantum_solver import load_and_prepare_data, solve_with_multilevel_quantum, get_route_cost_and_feasibility, solve_tsptw_with_insertion
from forecaster import generate_demand_predictions
from qiskit_aer import AerSimulator
import numpy as np
import uvicorn

def clean_data_for_json(data):
    """
    Recursively iterate through the dictionary/list structure and convert NumPy/Pandas types to native Python types.
    """
    if isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, pd.Series):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict('records')  # Convert to list of dicts for JSON compatibility
    else:
        return data

app = FastAPI(title="Quantum Carbon Route Solver", description="FastAPI backend for quantum and classical carbon route solvers.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic models for request bodies
class SolveRequest(BaseModel):
    file_path: str = "c101.csv"
    vehicle_capacity: int = 200
    p: int = 5
    T: int = 10
    lam1: float = 0.4
    quadratic_weight: float = 1.0
    subproblem_size_limit: int = 22
    cluster_radius: float = 15.0

def get_customer_locations_df(num_customers: int = 100, single_location: bool = True):
    """Create sample customer locations DataFrame. Assume single location for simplicity."""
    if single_location:
        location_id = 1
        customer_ids = list(range(1, num_customers + 1))
        locations = [location_id] * num_customers
    else:
        # If needed, assign random locations, but keep simple for now
        customer_ids = list(range(1, num_customers + 1))
        locations = [1] * num_customers  # Still single location

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'location_id': locations
    })
    return df

def classical_baseline_solve(params):
    """Classical heuristic solver - greedy insertion similar to quantum step."""
    unserved_customers = set(range(1, params["total_customers"] + 1))
    final_routes = []

    while unserved_customers:
        # Find best customer to add based on distance to depot
        best_customer = min(unserved_customers, key=lambda c: params['distance_matrix'][0, c])

        if params['demands'][best_customer] > params["vehicle_capacity"]:
            break  # Should not happen

        # Try to insert into existing routes or start new one
        inserted = False
        for route in final_routes:
            if sum(params['demands'][c] for c in route if c != 0) + params['demands'][best_customer] <= params["vehicle_capacity"]:
                _, new_route = solve_tsptw_with_insertion(route + [best_customer], params)
                if new_route:
                    final_routes[final_routes.index(route)] = new_route
                    inserted = True
                    break

        if not inserted:
            # Start new route
            new_route = [0, best_customer, 0]
            final_routes.append(new_route)

        unserved_customers.remove(best_customer)

    # Calculate total cost
    total_cost = sum(get_route_cost_and_feasibility(r, params)[0] for r in final_routes if get_route_cost_and_feasibility(r, params)[0] is not None)

    return final_routes, total_cost


from qiskit_aer import AerSimulator # Ensure this is imported at the top of your main.py

@app.post("/solve/quantum-hybrid")
async def solve_quantum_hybrid():
    # Load base data and generate predictions (Keep this section)
    params = load_and_prepare_data("c101.csv", 200)
    customer_locations_df = get_customer_locations_df(params['total_customers'])
    predictions = generate_demand_predictions(customer_locations_df)
    for cust_id in predictions:
        customer_idx = int(cust_id)
        if customer_idx < len(params['demands']):
            params['demands'][customer_idx] = predictions[cust_id]

    # Solve with quantum hybrid (Long calculation)
    backend = AerSimulator()
    # Note: total_cost will be a numpy type (e.g., numpy.float64)
    routes, total_cost = solve_with_multilevel_quantum(
        params=params, p=5, T=10, lam1=0.4, quadratic_weight=1.0, 
        backend=backend, subproblem_size_limit=22, cluster_radius=15.0
    )

    # --- FIX 1: Handle Solver Failure (NoneType Check) ---
    if routes is None:
        return {
            "solver": "quantum-hybrid",
            "total_vehicles": 0,
            "total_cost": 0.0,
            "vehicles": [],
        }
    # -----------------------------------------------------

    vehicle_details = []
    for i, route in enumerate(routes):
        # FIX: Ensure route elements are simple Python ints for get_route_cost_and_feasibility
        route_list = [int(c) for c in route]

        cost, time = get_route_cost_and_feasibility(route_list, params)
        demand = sum(params['demands'][c] for c in route_list if c != 0)

        # Create route_coordinates list
        route_coordinates = []
        for customer_id in route_list:
            df_row = params['dataframe'].iloc[int(customer_id)]
            lat = df_row['YCOORD.']
            lon = df_row['XCOORD.']
            route_coordinates.append([float(lat), float(lon)])

        vehicle_details.append({
            "vehicle_id": int(i + 1), # FIX 2a: Convert all numbers to standard Python types
            "route": route_list,      # FIX 2b: Use the standard Python list
            "cost": float(cost),      # FIX 2c: Convert cost to float
            "time": float(time),      # FIX 2d: Convert time to float
            "demand": float(demand),   # FIX 2e: Convert demand to float
            "route_coordinates": route_coordinates
        })

    # Create route_coordinates list of dictionaries
    route_coordinates = []
    for route in routes:
        coords_for_route = []
        for customer_id in route:
            if customer_id != 0:  # Skip depot
                row = params['dataframe'].iloc[int(customer_id)]
                lat = row['YCOORD.']
                lon = row['XCOORD.']
                coords_for_route.append([float(lat), float(lon)])
        route_coordinates.append({"coordinates": coords_for_route})

    result = {
        "solver": "quantum-hybrid",
        "total_vehicles": int(len(routes)), # FIX 2f: Ensure length is cast to int
        "total_co2": float(total_cost),     # FIX 2g: Convert total_cost to float
        "vehicles": vehicle_details,
        "route_coordinates": route_coordinates
    }
    return clean_data_for_json(result)
@app.post("/solve/classical-baseline")
async def handle_classical_solve():
    # ... your existing data loading logic ...
    params = load_and_prepare_data('c101.csv', 200)
    customer_locations_df = get_customer_locations_df(params['total_customers'])
    predictions = generate_demand_predictions(customer_locations_df)
    for cust_id in predictions:
        customer_idx = int(cust_id)
        if customer_idx < len(params['demands']):
            params['demands'][customer_idx] = predictions[cust_id]

    customers_to_route = list(range(1, 101))

    # total_cost contains the cost (float/int)
    # single_route_list contains ONE SINGLE route list: [0, 20, 21, ..., 0]
    total_cost, single_route_list = solve_tsptw_with_insertion(customers_to_route, params)

    # --- START OF FIX ---
    if single_route_list is None:
        # If the solver failed, return the empty result (safety check)
        return {"routes": [], "total_co2": 0, "vehicles": 0}

    # FIX: Wrap the single route list in a list of lists (required for the subsequent loop).
    final_routes = [single_route_list]

    # --- END OF FIX ---

    # Now construct vehicles array like quantum
    vehicle_details = []
    for i, route in enumerate(final_routes):
        route_list = [int(c) for c in route]

        cost, time = get_route_cost_and_feasibility(route_list, params)
        demand = sum(params['demands'][c] for c in route_list if c != 0)

        # Create route_coordinates list
        route_coordinates = []
        for customer_id in route_list:
            df_row = params['dataframe'].iloc[int(customer_id)]
            lat = df_row['YCOORD.']
            lon = df_row['XCOORD.']
            route_coordinates.append([float(lat), float(lon)])

        vehicle_details.append({
            "vehicle_id": int(i + 1),
            "route": route_list,
            "cost": float(cost),
            "time": float(time),
            "demand": float(demand),
            "route_coordinates": route_coordinates
        })

    py_total_cost = float(total_cost)

    result = {
        "solver": "classical-baseline",
        "total_vehicles": int(len(final_routes)),
        "total_co2": py_total_cost,
        "vehicles": vehicle_details
    }
    return clean_data_for_json(result)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
