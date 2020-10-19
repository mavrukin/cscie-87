from functools import partial
import numpy


def beta_modifier(beta_scale, day, beta):
    return beta if day <= 31 else beta * beta_scale


def model_sir(
    population_size,
    beta,
    gamma,
    simulation_days,
    vaccination_factor,
    beta_modifier = None,
    debug = False):
    infected = numpy.empty(simulation_days)
    susceptible = numpy.empty(simulation_days)
    removed = numpy.empty(simulation_days)
    infected[0] = 1 / population_size
    susceptible[0] = 1.0 - infected[0]
    removed[0] = 0
    peak_infected = 0
    peak_infected_day = 0
    for day in range(1, simulation_days):
        modified_beta = beta
        if beta_modifier is not None:
            modified_beta = beta_modifier(day, beta)
        if debug:
            print(f"SIR on {day - 1}: S(n) + I(n) + R(n) = {susceptible[day - 1]} + {infected[day - 1]} + {removed[day - 1]} = {susceptible[day - 1] + infected[day - 1] + removed[day - 1]}")
        susceptible[day] = susceptible[day - 1] - (infected[day - 1] * modified_beta * susceptible[day - 1]) - vaccination_factor
        infected[day] = infected[day - 1] + (infected[day - 1] * modified_beta * susceptible[day - 1]) - (infected[day - 1] * gamma)
        removed[day] = removed[day - 1] + (infected[day - 1] * gamma) + vaccination_factor
        if infected[day] > peak_infected:
            peak_infected = infected[day]
            peak_infected_day = day
    if debug:
        print(f"SIR on {simulation_days}: S(n) + I(n) + R(n) = {susceptible[simulation_days - 1]} + {infected[simulation_days - 1]} + {removed[simulation_days - 1]} = {susceptible[simulation_days - 1] + infected[simulation_days - 1] + removed[simulation_days - 1]}")
    return peak_infected, peak_infected_day


def process_sim_model():
    population_size = 120000
    beta = 0.2
    days_to_recover = 20
    gamma = (1 / days_to_recover)
    simulation_days = 365
    model_a_result = model_sir(population_size, beta, gamma, simulation_days, 0)

    print()
    print(f"Final SIR - Model A")
    print(f"Peak Infections: {model_a_result[0]} on {model_a_result[1]}")

    for vaccine_factor in [0.0025, 0.005, 0.01, 0.02, 0.04]:
        model_b_result = model_sir(population_size, beta, gamma, simulation_days, vaccine_factor)
        print(f"Final SIR - Model B[{vaccine_factor}]")
        print(f"Peak Infections: {model_b_result[0]} on {model_b_result[1]}")

    model_c_result = model_sir(population_size, beta, gamma * 2, simulation_days, 0)
    print(f"Final SIR - Model C")
    print(f"Peak Infections: {model_c_result[0]} on {model_c_result[1]}")

    for beta_ in [1.5, 1.7, 2.0, 2.5, 3.0]:
        model_d_result = model_sir(population_size, beta, gamma, simulation_days, 0, partial(beta_modifier, 1 / beta_))
        print(f"Final SIR - Model D[{beta_}]")
        print(f"Peak Infections: {model_d_result[0]} on {model_d_result[1]}")


if __name__ == "__main__":
    process_sim_model()