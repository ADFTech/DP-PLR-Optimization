import pandas as pd
import numpy as np
import random
import pickle as p
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from deap import base, creator, tools, gp
import holidays
import warnings

warnings.filterwarnings("ignore")

# Malaysia holidays
MY_holidays = holidays.MY(subdiv="KUL", years=range(2021, 2049))

# Read CSV
df_CL = pd.read_csv(r'forecasted_weather_HLX_OPT_New.csv', header=0)
df_CL['timestamp'] = pd.to_datetime(df_CL['timestamp'], format='mixed', errors='coerce')

df_CL.set_index('timestamp', inplace=True)
df_CL_resampled = df_CL.resample('15min').interpolate()
df_CL_resampled.reset_index(inplace=True)
df_CL_F = df_CL_resampled

df_CL_F["weekday"] = df_CL_F['timestamp'].dt.dayofweek > 4
df_CL_F['holiday'] = df_CL_F.timestamp.isin(MY_holidays)
df_CL_F.replace({False: 0, True: 1}, inplace=True)

df_CL_F['year'] = df_CL_F.timestamp.apply(lambda x: x.year)
df_CL_F['month'] = df_CL_F.timestamp.apply(lambda x: x.month)
df_CL_F['day'] = df_CL_F.timestamp.apply(lambda x: x.day)
df_CL_F['hour'] = df_CL_F.timestamp.apply(lambda x: x.hour)
df_CL_F['minute'] = df_CL_F.timestamp.apply(lambda x: x.minute)
df_CL_F.drop(columns=['timestamp'], inplace=True)

# Load Models
cl_mdl_file = open('Model/hlx_gbt_cl_model.p', 'rb')
P_mdl_file = open('Model/hlx_gbt_kwh_model.p', 'rb')

cl_mdl = p.load(cl_mdl_file)
p_mdl = p.load(P_mdl_file)

DP_low = 0.1
DP_high = 0.25



Cl_features=['temperature','humidity','weekday','holiday','year','month','day','hour','minute']


P_features =   ['DP','Cooling_Load','kW_RT','Year','Month','Day','Hour','Minute']



X_CL = df_CL_F[Cl_features]


def forecast(inp_x_arr):
    x_arr = np.array(inp_x_arr[0] + list(X_p_next[1:]))
    return p_mdl.predict(x_arr.reshape(1, -1))


def set_vals():
    return [random.uniform(DP_low, DP_high)]


def prep_new_X_vals(X_cl_hist, next_time):
    print(X_cl_hist)
    temp = X_cl_hist.iloc[-1, 0]
    humidity = X_cl_hist.iloc[-1, 1]
    kW_RT = random.uniform(0.5, 0.65)
    # temp = random.uniform(83, 96)
    # humidity = random.uniform(80, 100)
    Month = next_time.month
    Day = next_time.day
    Hour = next_time.hour
    Minute = next_time.minute
    Year = 2025
    load_pred = cl_mdl.predict(X_cl_hist)
    if isinstance(load_pred, (list, np.ndarray)):
        load_pred = float(np.ravel(load_pred)[-1])
    return np.array([0, load_pred, kW_RT, temp,humidity, Year, Month, Day, Hour, Minute], dtype=float)


# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("set_vals", set_vals)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.set_vals, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", forecast)
toolbox.register("mate", tools.cxUniform, indpb=0.25)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=3)


def check_pop_vals(varlis, Pred_Cl, expected_next_power):
    DP = varlis[0][0]
    if ((Pred_Cl <=380 and 0.075<=DP<=0.125) or (Pred_Cl>380 and 0.15<=DP<=0.25)):
        return True
    else:
        return False


def optimize(n_pop, n_gen):
    print("Starting optimization...")

    # Define probabilities for crossover and mutation
    CXPB = 0.5  # crossover probability
    MUTPB = 0.2  # mutation probability

    # Create population
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    for gen in range(n_gen):
        offspring = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = list(map(toolbox.clone, offspring))

        # ---------------- SAFE CROSSOVER ----------------
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # ---------------- SAFE MUTATION ----------------
        for idx in range(len(offspring)):
            if random.random() < MUTPB:
                ind = offspring[idx]

                # Normalize input: make sure it's always an Individual list
                if isinstance(ind, (float, int)):
                    ind = creator.Individual([ind])
                elif not isinstance(ind, creator.Individual):
                    # If somehow it's another list type, wrap it safely
                    ind = creator.Individual(list(ind))

                # Mutate safely
                try:
                    toolbox.mutate(ind)
                except Exception as e:
                    print(f"Mutation error for individual {ind}: {e}")
                    continue

                # Replace in offspring
                offspring[idx] = ind
                del offspring[idx].fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select next generation
        pop[:] = offspring

        # Gather stats
        record = stats.compile(pop)
        hof.update(pop)
        # print(f"Gen {gen+1}/{n_gen} | Min: {record['min']:.4f} | Max: {record['max']:.4f}")

    best_ind = hof[0]
    mean_pred = np.mean([ind.fitness.values[0] for ind in pop])
    return best_ind, mean_pred


def CH_OPT():
    n_pop = 70
    n_gen = 30

    global X_p_next, X_CL_Next, Pred_Cl, expected_next_kwh, next_datetime

    size = len(X_CL)
    start_point = 0
    results = []
    Nsteps = 200

    while start_point < size - 1:
        for i in tqdm(range(Nsteps)):
            idx = start_point + i
            if idx >= size - 1:
                break

            X_CL_Next = X_CL.iloc[: idx + 1, :]
            year = 2025
            Month_curr = X_CL_Next.iloc[-1, -4]
            Day_curr = X_CL_Next.iloc[-1, -3]
            hour_curr = X_CL_Next.iloc[-1, -2]
            minute_curr = X_CL_Next.iloc[-1, -1]
            curr_datetime = datetime(year, int(Month_curr), int(Day_curr), int(hour_curr), int(minute_curr))
            next_datetime = curr_datetime + timedelta(minutes=15)

            X_p_next = prep_new_X_vals(X_CL_Next, next_datetime)
            Pred_Cl = X_p_next[1]
            expected_next_kwh = abs(forecast([set_vals()]))

            print('predicted_kwh', expected_next_kwh)

            office_hour_start = 7.0
            office_hour_end = 21.0

            if (next_datetime.hour >= office_hour_end) or (next_datetime.hour <= office_hour_start):
                perf = 0.
                opt = 0
                best = [(0, 0)]
                office = False
            else:
                best, mean_pred = optimize(n_pop, n_gen)
                if mean_pred != 0.:
                    opt = forecast(best)[0]
                    perf = (abs(abs(mean_pred) - abs(opt)) / abs(mean_pred)) * 100
                else:
                    opt = 0
                    perf = 0
                office = True

            next_time_string = next_datetime.strftime("%Y-%m-%d %H:%M:%S")
            if office:
                print(f"The optimized values at {next_time_string} with an output Consumption of {opt} kWh \n DP:{best[0][0]}")
                print(X_p_next)
            else:
                print(f"{next_time_string}: Non-office hour, optimization has been stopped.")

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("_______________________its DONEEEEEEEEE!!!!!_____________________")

            results.append({
                "timestamp": next_datetime,
                "idx": idx,
                "Pred_Cl": Pred_Cl,
                # "expected_power": expected_next_power,
                "optimized_kwh": opt,
                "DP": best[0][0] if best != [(0, 0)] else 0,
                "performance_%": perf
            })

        start_point += Nsteps

    df_results = pd.DataFrame(results)
    df_results.to_csv("DP_Optimization_HLX_Result.csv", index=False)
    print("Optimization completed and saved to 'DP_Optimization_HLX_Result.csv'")
    return df_results


# Run main optimization
CH_OPT()

