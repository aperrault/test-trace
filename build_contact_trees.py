import load_contact_data
import numpy as np
import numpy.random
import scipy
import scipy.stats
import load_contact_data
import fit_dist
import code
import matplotlib
import matplotlib.pyplot as plt
from numba import jit, njit
from numba.typed import List
from numba import prange
import numba
from multiprocessing import Pool
import math
import random
import scipy.optimize
import json

age_intervals = [(0, 14), (15, 24), (25, 54), (55, 64), (65, np.inf)]
age_specific_IFRs = np.array([0.0001, 0.0001, 0.0013, 0.007, 0.068])
age_vector_US = np.array([0.1862, 0.1312, 0.3929, 0.1294, 0.1603])
age_vector_US = age_vector_US / np.sum(age_vector_US)
false_positive_rate = 0.0
frac_asymptoms = 0.20
asymp_infectiousness = 0.35
presymp_infectiousness = 0.63
""" relative transmission rate of asymptoms individual """
days_infectious_before_symptoms = 2
days_infectious_after_symptoms = 5
total_infectious_days = days_infectious_before_symptoms + days_infectious_after_symptoms
asymp_quarantine_length = 14
symp_quarantine_length = 10
symptoms_household_SAR = 0.2
symptoms_external_SAR = 0.06
superspreader_SAR_ratio = 10
# frac_infs_from_SS = 0.9
index_R0 = 2.548934288671604
raw_R0 = 2.5
# raw_R0 = 1.48 # 50% reduction
quarantine_dropout_rate_default = 0.05
length_of_sympts_to_track = 30
symptom_false_positive_chance = 0.01
pooling_max = 20
dropout_reduction_for_symptoms = 0.5
frac_self_isolate = 0.9

frac_SS_cal = 1 / (1 + np.exp(2.1))
base_household_cal = 1 / (1 + np.exp(3.3))
base_external_cal = 1 / (1 + np.exp(5.))
SS_mult_cal = 1 + np.log(1 + np.exp(23.4))
phys_mult_cal = 1 + np.log(1 + np.exp(4.2))

fraction_household_physical = [0.] # filled in by get_contacts_per_age
fraction_external_physical = [0.] # filled in by get_contacts_per_age

n_processes = 1

def distancing_from_R0_squared(R0_squared_obs):
    # return 1 - math.sqrt(1 / R0_squared_obs)
    return 1 - raw_R0 / R0_squared_obs

def reduction_from_R0_squared(R0_squared_obs):
    return 1 - (R0_squared_obs / index_R0) / raw_R0

""" convert raw numeric age into age interval """
def raw_age_to_interval(age, age_intervals):
    for i in range(len(age_intervals)):
        if age >= age_intervals[i][0] and age <= age_intervals[i][1]:
            return i

""" process POLYMOD data into dict of contacts by age """
def get_contacts_per_age():
    total_household = 0
    total_external = 0
    age_to_id_dict = {}
    age_to_contacts_dict = {}
    for i in range(len(age_intervals)):
        age_to_contacts_dict[i] = List()
        age_i = np.logical_and(load_contact_data.raw_participants["part_age"]
                               >= age_intervals[i][0], load_contact_data.raw_participants["part_age"] <= age_intervals[i][1])
        age_to_id_dict[i] = list(load_contact_data.raw_participants[age_i]['part_id'])
        for j in age_to_id_dict[i]:
            contacts_j = load_contact_data.raw_contacts[load_contact_data.raw_contacts['part_id'] == j]
            contacts = List()
            contacts.append((1, True, True, True, True))
            contacts.pop()
            """ contact format: (age of contact, gender, is_household, is_daily, duration, is_physical) """
            contacts_numpy = contacts_j.to_numpy()
            for row in range(np.shape(contacts_numpy)[0]):
                if contacts_numpy[row][14] <= 2:
                    continue
                # assert(contacts_numpy[row][5] == 'F' or contacts_numpy[row][5] == 'M')
                try:
                    contacts.append((raw_age_to_interval(int(contacts_numpy[row][2]), age_intervals),
                                     contacts_numpy[row][5] == 'F',
                                     bool(contacts_numpy[row][6]),
                                     int(contacts_numpy[row][12]) == 1,
                                     int(contacts_numpy[row][13]) == 1))
                    if bool(contacts_numpy[row][6]):
                        total_household += 1
                        if int(contacts_numpy[row][12]) == 1:
                            fraction_household_physical[0] += 1
                    else:
                        total_external += 1
                        if int(contacts_numpy[row][12]) == 1:
                            fraction_external_physical[0] += 1
                except Exception:
                    pass
            age_to_contacts_dict[i].append(contacts)
    fraction_external_physical[0] = fraction_external_physical[0] / total_external
    fraction_household_physical[0] = fraction_household_physical[0] / total_household
    return age_to_contacts_dict

# @njit
# def true_positive_test_rate(test_time_rel_to_symptoms):
#     if test_time_rel_to_symptoms <= -3:
#         return 0.0
#     if test_time_rel_to_symptoms == -2:
#         return 0.0
#     if test_time_rel_to_symptoms == -1:
#         return 0.0
#     if test_time_rel_to_symptoms == 0:
#         return 0.0
#     else:
#         return 0.0

@njit
def false_negative_test_rate(test_time_rel_to_symptoms):
    if test_time_rel_to_symptoms <= -3:
        return 1.0
    if test_time_rel_to_symptoms == -2:
        return 0.95
    if test_time_rel_to_symptoms == -1:
        return 0.7
    if test_time_rel_to_symptoms == 0:
        return 0.4
    if test_time_rel_to_symptoms <= 4:
        return 0.25
    else:
        return min(0.25 + (test_time_rel_to_symptoms - 4) * 0.0375, 1.)


@jit(nopython=True, parallel=True)
def calc_test_results(num_individuals, test_time_rel_to_symptoms, I_COVID, seed=0):
    test_results = np.zeros(num_individuals)
    for i in prange(num_individuals):
        test_results[i] = calc_test_single(test_time_rel_to_symptoms[i], I_COVID[i])
    return test_results

@njit
def calc_test_single(test_time_rel_to_symptoms, I_COVID):
    if I_COVID:
        return np.random.binomial(n=1, p=1 - false_negative_test_rate(int(round(test_time_rel_to_symptoms))))
    else:
        return 0

""" draw properties of seed index cases """
def draw_seed_index_cases(num_individuals, age_vector, cases_contacted=1.0,
                          t_exposure=None, fpr=false_positive_rate, initial=False, skip_days=False,
                          random_testing=False, test_freq=0., test_delay=0.,
                          frac_SS=frac_SS_cal, frac_vacc=0., seed=0):
    np.random.seed(seed)
    if t_exposure is None:
        t_exposure = np.zeros(num_individuals)
    params = {}
    n_ages = scipy.stats.multinomial.rvs(num_individuals, age_vector)
    if t_exposure is None:
        params['t_exposure'] = np.zeros(num_individuals)
    else:
        params['t_exposure'] = t_exposure
    params['t_last_exposure'] = params['t_exposure']
    params['I_contacted'] = scipy.stats.bernoulli.rvs(p=cases_contacted, size=num_individuals)
    params['n_age'] = np.sum([[i] * n_ages[i] for i in range(len(n_ages))])
    params['I_COVID'] = scipy.stats.bernoulli.rvs(p=1 - fpr, size=num_individuals)
    if not initial:
        params['I_asymptoms'] = scipy.stats.bernoulli.rvs(p=frac_asymptoms, size=num_individuals) * params['I_COVID']
    else:
        params['I_asymptoms'] = np.zeros(num_individuals)
    params['I_symptoms'] = 1 - params['I_asymptoms']
    if not initial:
        params['I_vacc'] = scipy.stats.bernoulli.rvs(p=frac_vacc, size=num_individuals)
    else:
        params['I_vacc'] = np.zeros(num_individuals) # TODO: HACK
    params['t_incubation'] = np.maximum(scipy.stats.lognorm.rvs(s=0.65, scale=np.exp(1.57), size=num_individuals), 0) + t_exposure
    params['t_incubation_infect'] = np.maximum(params['t_incubation'] - days_infectious_before_symptoms, 0)
    if initial:
        params['I_self_isolate'] = np.ones(num_individuals) * params['I_symptoms']
    else:
        params['I_self_isolate'] = scipy.stats.bernoulli.rvs(p=frac_self_isolate, size=num_individuals) * params['I_symptoms']
    params['t_false_positive'] = np.random.geometric(p=symptom_false_positive_chance, size=num_individuals) + t_exposure
    params['t_self_isolate'] = scipy.stats.gamma.rvs(loc=days_infectious_before_symptoms, scale=1.22, a=1/0.78, size=num_individuals) + params['t_incubation_infect']
    if skip_days:
        params['n_transmission_days'] = (
            params['I_asymptoms'] * total_infectious_days +
            params['I_symptoms'] * (1 - params['I_self_isolate']) * total_infectious_days +
            params['I_symptoms'] * params['I_self_isolate'] * (params['t_self_isolate'] - params['t_incubation_infect']))
        params['n_quarantine_days'] = np.zeros(num_individuals)
        params['n_isolation_days'] = params['I_self_isolate'] * symp_quarantine_length
        params['n_tests'] = np.zeros(num_individuals)
    else:
        (params['n_quarantine_days'], params['n_transmission_days'],
         params['n_isolation_days'], params['n_tests'],
         params['n_monitoring_days'], params['n_false_positive_isolation_days'],
         params['n_billed_tracer_days']) = get_transmission_days(
            t_exposure=params['t_exposure'], t_last_exposure=params['t_last_exposure'],
            I_isolation=params['I_self_isolate'], t_isolation=params['t_self_isolate'],
            t_infectious=params['t_incubation_infect'],
            I_symptoms=params['I_symptoms'], t_symptoms=params['t_incubation'],
            I_random_testing=random_testing, test_freq=test_freq,
            test_delay=test_delay, t_false_positive=params['t_false_positive'])
    params['id_original_case'] = (np.ones(num_individuals) * -1).astype(int)
    params['id_infected_by'] = (np.ones(num_individuals) * -1).astype(int)
    params['I_superspreader'] = scipy.stats.bernoulli.rvs(p=frac_SS, size=num_individuals) * params['I_COVID']
    if initial:
        params['I_test_positive'] = np.ones(num_individuals)
    else:
        params['I_test_positive'] = calc_test_results(num_individuals,
                                                     params['t_self_isolate'] - params['t_incubation_infect'],
                                                     params['I_COVID'], seed=seed)
    params['I_infected_by_presymp'] = np.ones(num_individuals) * -1
    params['I_infected_by_asymp'] = np.ones(num_individuals) * -1
    return params

@njit
def draw_contacts(contact_days, n_transmission_days, t_incubation_infect, index, base_reduction=0., get_dummy=False, frac_vacc=0.,
                  seed=0):
    np.random.seed(seed)
    contact_list = List()
    contacts = List()
    selection = np.random.choice(len(contact_days))
    for day in range(int(n_transmission_days)):
        contacts.append(contact_days[selection])
    for day in range(int(n_transmission_days)):
        for (i, contact) in enumerate(contacts[day]):
            # if (day == 0 or not contact[3] or np.random.binomial(n=1, p=(1. - 5./7.) ** day)) or get_dummy:
            # if (day == 0 or not contact[3] or np.random.binomial(n=1, p=0.5)) or get_dummy:
            if day == 0 or not contact[3] or get_dummy:
                if not get_dummy:
                    if not contact[2] and not contact[3]:
                        if not np.random.binomial(n=1, p=1 - base_reduction):
                            continue
                """ contact format: (age of contact, gender, is_household, is_daily,
                                     t_exposure, index of infector, t_last_exposure, is_physical, contact_is_vacc) """
                contact_ext = np.zeros(9)
                contact_ext[0] = contact[0]
                contact_ext[1] = contact[1]
                contact_ext[2] = contact[2]
                contact_ext[3] = contact[3]
                if contact[3]:
                    contact_days_mask = np.random.binomial(n=1, p=1 - base_reduction, size=int(n_transmission_days))
                    if np.max(contact_days_mask) < 0.01:
                        continue
                    infection_days = np.where(contact_days_mask > 0.99)[0]
                    day_of_infection = infection_days[np.random.randint(len(infection_days))]
                    contact_ext[6] = t_incubation_infect + np.max(infection_days)
                else:
                    day_of_infection = day
                    contact_ext[6] = day + t_incubation_infect
                contact_ext[4] = day_of_infection + t_incubation_infect
                contact_ext[5] = index
                contact_ext[7] = contact[4]
                contact_ext[8] = np.random.binomial(n=1, p=frac_vacc)
                contact_list.append(contact_ext)
    return contact_list

@njit
def draw_all_contacts(contacts_by_age, n_transmission_days, t_incubation_infect, num_individuals, base_reduction=0.0,
                      frac_vacc=0., seed=0):
    np.random.seed(seed)
    contact_dict = List()
    for i in range(num_individuals):
        a = draw_contacts(contact_days=contacts_by_age[0],
                          n_transmission_days=n_transmission_days[0],
                          t_incubation_infect=t_incubation_infect[0],
                          index=i, frac_vacc=frac_vacc, get_dummy=True)
        if len(a) > 0:
            break
    for i in range(num_individuals):
        contact_dict.append(a)
    for i in range(num_individuals):
        contact_dict[i] = draw_contacts(contact_days=contacts_by_age[i],
                                        n_transmission_days=n_transmission_days[i],
                                        t_incubation_infect=t_incubation_infect[i],
                                        frac_vacc=frac_vacc,
                                        index=i, base_reduction=base_reduction)
    return contact_dict

""" draw index cases assuming uniform attack rate. draw contacts from POLYMOD """
def draw_contact_generation(index_cases, base_reduction=0.0, frac_vacc=0.0, seed=0):
    num_individuals = len(index_cases['I_COVID'])
    if num_individuals == 0:
        return []
    contacts_by_age = [age_to_contact_dict[index_cases['n_age'][i]] for i in range(num_individuals)]
    contact_dict = draw_all_contacts(contacts_by_age=contacts_by_age,
                                     n_transmission_days=index_cases['n_transmission_days'],
                                     t_incubation_infect=index_cases['t_incubation_infect'],
                                     num_individuals=num_individuals,
                                     base_reduction=base_reduction,
                                     frac_vacc=frac_vacc,
                                     seed=seed)
    return contact_dict

@jit(nopython=True, parallel=True)
def fill_QII(t_exposure, t_last_exposure, trace_delay, test_delay, t_self_isolate,
             I_self_isolate, t_incubation_infect,
             t_incubation, I_symptoms, I_household, t_false_positive,
             id_infected_by, num_g1_cases, g0_I_self_isolate, g0_t_self_isolate,
             g0_I_test_positive, g0_I_symptoms, g0_I_contacted,
             g0_I_superspreader, false_negative_traces, trace=False, ttq=False,
             quarantine_by_parent_case_release=False,
             early_release_by_parent_case=np.zeros(1), early_release=False,
             wait_before_testing=0, wait_until_testing=0,
             ttq_double=False, dropouts=np.zeros(1).reshape(1, -1),
             precalc_dropout=0,
             monitor=False, seed=0, hold_hh=False,
             quarantine_dropout_rate=quarantine_dropout_rate_default):
    np.random.seed(seed)
    n_quarantine_days = np.zeros(num_g1_cases)
    n_transmission_days = np.zeros(num_g1_cases)
    n_isolation_days = np.zeros(num_g1_cases)
    n_monitoring_days = np.zeros(num_g1_cases)
    n_billed_tracer_days = np.zeros(num_g1_cases)

    n_tests = np.zeros(num_g1_cases)
    n_false_positive_isolation_days = np.zeros(num_g1_cases)

    secondary_cases_traced = 0
    secondary_cases_monitored = 0

    I_bidirectionally_traced = np.zeros(len(g0_I_self_isolate))
    earliest_self_isolates = np.ones(len(g0_I_self_isolate)) * 10000
    # if bidirectional_SS:
    #     for i in prange(num_g1_cases):
    #         parent_case = int(id_infected_by[i])
    #         if not g0_I_test_positive[parent_case] and g0_I_symptoms[parent_case] and I_self_isolate[i]:
    #             if not I_bidirectionally_traced[parent_case]:
    #                 I_bidirectionally_traced[parent_case] = 1
    #                 earliest_self_isolates[parent_case] = t_self_isolate[i]
    #             earliest_self_isolates[parent_case] = min(t_self_isolate[i], earliest_self_isolates[parent_case])
    #     for i in prange(len(g0_I_self_isolate)):
    #         if I_bidirectionally_traced[i] and not g0_I_self_isolate[i]:
    #             g0_I_test_positive[i] = calc_test_single(earliest_self_isolates[i], True)

    # if ttq:
    #     for i in prange(num_g1_cases):
    #         parent_case = int(id_infected_by[i])
    #         if g0_I_test_positive[parent_case]:
    #             I_test_positive[i] = calc_test_single(g0_t_self_isolate[parent_case] + trace_delay - t_incubation[i], True)
    for i in prange(num_g1_cases):
        """ did the parent case test positive? """
        parent_case = int(id_infected_by[i])
        if (g0_I_test_positive[parent_case] and g0_I_self_isolate[parent_case] and g0_I_contacted[parent_case]
            and trace and not false_negative_traces[i]):
            secondary_cases_traced += 1
            t_quarantine_start = g0_t_self_isolate[parent_case] + trace_delay + test_delay
            I_test = False
            t_test_day = -1
            t_quarantine_end = max(t_last_exposure[i] + asymp_quarantine_length, t_quarantine_start)
            if early_release and quarantine_by_parent_case_release[parent_case] and not ttq and not (hold_hh and I_household[i]):
                t_quarantine_end = min(t_quarantine_end, t_quarantine_start + early_release_by_parent_case[parent_case])
            if ttq:
                if not early_release:
                    I_test = True
                    t_test_day = max(t_quarantine_start + wait_before_testing, t_last_exposure[i] + wait_until_testing + wait_before_testing)
                elif early_release and quarantine_by_parent_case_release[parent_case] and not (hold_hh and I_household[i]):
                    I_test = True
                    t_test_day = t_quarantine_start + early_release_by_parent_case[parent_case]
                else:
                    I_test = False
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=t_quarantine_start,
                                     t_quarantine_end=t_quarantine_end,
                                     I_quarantine=True,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_false_positive=t_false_positive[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     I_monitor=monitor,
                                     t_monitor=t_quarantine_start,
                                     t_monitor_end=t_quarantine_end,
                                     I_test=I_test,
                                     t_test_day=t_test_day,
                                     test_delay=test_delay,
                                     I_double_test=ttq_double,
                                     wait_before_testing=wait_before_testing,
                                     I_early_release=early_release,
                                     early_release_day=t_quarantine_start + early_release_by_parent_case[parent_case],
                                     dropouts=dropouts[i].reshape(1, -1),
                                     precalc_dropout=precalc_dropout,
                                     quarantine_dropout_rate=quarantine_dropout_rate)
        else:
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=0,
                                     I_quarantine=False,
                                     t_quarantine_end=0,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_false_positive=t_false_positive[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     I_monitor=False,
                                     t_monitor=-1,
                                     I_test=False,
                                     t_test_day=-1,
                                     test_delay=test_delay,
                                     quarantine_dropout_rate=quarantine_dropout_rate)
        n_quarantine_days[i] += tmp[0]
        n_isolation_days[i] += tmp[1]
        n_transmission_days[i] += tmp[2]
        n_tests[i] += tmp[3]
        n_monitoring_days[i] += tmp[4]
        n_false_positive_isolation_days[i] += tmp[5]
        n_billed_tracer_days[i] += tmp[6]
    return (n_quarantine_days, n_transmission_days, n_isolation_days, secondary_cases_traced, secondary_cases_monitored, n_monitoring_days,
            n_tests, n_false_positive_isolation_days, n_billed_tracer_days)

# positives = [0]
# negatives = [0]
# days_rel_to_sympts = []

@njit
def calculate_QII_days(t_exposure, t_last_exposure,
                       t_infectious, t_false_positive,
                       I_quarantine=False, t_quarantine=0., t_quarantine_end=0.,
                       I_isolation=False, t_isolation=0.,
                       I_symptoms=False, t_symptoms=0.,
                       I_monitor=False, t_monitor=0., t_monitor_end=0.,
                       I_test=False, t_test_day=0., test_delay=0.,
                       I_random_testing=False, test_freq=0.,
                       I_double_test=False, wait_before_testing=0,
                       early_release_day=0, I_early_release=False,
                       dropouts=np.zeros(1).reshape(1, -1), precalc_dropout=0,
                       quarantine_dropout_rate=quarantine_dropout_rate_default):
    # if I_random_testing and test_freq > 0.0:
    #     test_interval = 1 / test_freq
    # else:
    #     test_interval = 10000
    # first_test = t_infectious + np.random.uniform(0.0, test_interval)

    quarantine_days = 0
    isolation_days = 0
    transmission_days = 0
    monitoring_days = 0
    false_isolation_days = 0
    n_tests = 0

    t_isolation_end = t_isolation + symp_quarantine_length

    t_monitor_end = t_monitor + asymp_quarantine_length

    if I_quarantine and I_isolation:
        if t_quarantine_end > t_isolation_end:
            t_quarantine_end = t_isolation_end
    t_infectious_end = t_infectious + total_infectious_days

    day = t_exposure
    I_test_positive = False
    symptoms_observed = False
    tested = False
    last_test_result = True
    test_results_day_exists = False
    test_results_day = 0
    I_false_isolation = False
    t_false_isolation = t_exposure
    t_false_isolation_end = t_exposure
    billed_tracer_days = 0
    quarantine_dropout_chance = quarantine_dropout_rate
    quarantine_dropout_rate_pos_test = 0.0 * quarantine_dropout_rate
    quarantine_dropout_rate_not_released = quarantine_dropout_rate * (1 - dropout_reduction_for_symptoms)
    quarantine_dropout_rate_neg_test = quarantine_dropout_rate * 2

    # print("infected ", quarantine_dropout_chance)
    while not (day >= t_quarantine_end and day >= t_isolation_end and day >= t_isolation_end and day >= t_infectious_end and day >= t_false_isolation_end):
        # code.interact(local=locals())
        # symptoms are developed during quarantine or already exist when quarantine started
        if I_symptoms and day >= t_symptoms and ((I_quarantine and t_quarantine <= day < t_quarantine_end) or (I_monitor and t_monitor <= day < t_monitor_end)) and not symptoms_observed:
            if I_isolation:
                t_isolation = min(t_isolation, day + 1)
            else:
                t_isolation = day
                I_isolation = True
            t_isolation_end = t_isolation + symp_quarantine_length
            t_quarantine_end = min(t_quarantine_end, t_isolation_end)
            symptoms_observed = True

        # is this a billed tracer day
        if (I_quarantine and t_quarantine <= day < t_quarantine_end) or (I_monitor and t_monitor <= day < t_monitor_end):
            billed_tracer_days += 1

        # is this person quarantined today? do they drop out?
        if t_infectious <= day < t_infectious_end:
            if (not I_quarantine or not (t_quarantine <= day < t_quarantine_end)) and (not I_isolation or not (t_isolation <= day < t_isolation_end)):
                transmission_days += 1
            elif (I_isolation and t_isolation <= day < t_isolation_end) or (I_quarantine and t_quarantine <= day < t_quarantine_end) or (I_false_isolation and t_false_isolation <= day < t_false_isolation_end):
                isolation_days += 1
        elif I_quarantine and t_quarantine <= day < t_quarantine_end:
            if (I_isolation and t_isolation <= day < t_isolation_end):
                isolation_days += 1
            elif I_false_isolation and t_false_isolation <= day < t_false_isolation_end:
                false_isolation_days += 1
            else:
                quarantine_days += 1
        elif I_isolation and t_isolation <= day < t_isolation_end:
            isolation_days += 1
        elif I_false_isolation and t_false_isolation <= day < t_false_isolation_end:
            false_isolation_days += 1
        elif I_monitor and t_monitor <= day < t_monitor_end:
            monitoring_days += 1
            if day >= t_false_positive and not I_false_isolation:
                I_false_isolation = True
                t_false_isolation = day
                t_false_isolation_end = day + symp_quarantine_length

        # adjust quarantine dropout chance
        # print("day, dropout ", day, quarantine_dropout_chance)
        if I_early_release and early_release_day <= day < early_release_day + 1:
            quarantine_dropout_chance = quarantine_dropout_rate_not_released
        if test_results_day_exists and test_results_day <= day < test_results_day + 1:
            if I_test_positive:
                quarantine_dropout_chance = quarantine_dropout_rate_pos_test
            else:
                quarantine_dropout_chance = quarantine_dropout_rate_neg_test

        # drop out of quarantine
        if I_quarantine and t_quarantine <= day < t_quarantine_end and not (
            (I_isolation and t_isolation <= day < t_isolation_end) or
            (I_false_isolation and t_false_isolation <= day <= t_false_isolation_end)):

            if day - t_quarantine < precalc_dropout and quarantine_dropout_chance == quarantine_dropout_rate:
                if dropouts[0][int(day - t_quarantine)] > 0.:
                    t_quarantine_end = day
                    t_monitor_end = day
            elif np.random.binomial(n=1, p=quarantine_dropout_chance):
                t_quarantine_end = day
                t_monitor_end = day

        # is this person tested today?
        if I_double_test and day >= t_test_day and t_quarantine <= day < t_quarantine_end - test_delay:
            I_test_positive = calc_test_single(day - t_symptoms, True)
            test_results_day = day + test_delay
            test_results_day_exists = True
            tested = True
            n_tests += 1
            if not last_test_result and not I_test_positive:
                t_quarantine_end = min(day + test_delay, t_quarantine_end)
                I_double_test = False
            else:
                last_test_result = I_test_positive
                t_test_day = day + test_delay + wait_before_testing
        # print(I_test and day >= t_test_day and not tested and t_quarantine <= day < t_quarantine_end - test_delay)
        if I_test and day >= t_test_day and not tested and t_quarantine <= day < t_quarantine_end - test_delay:
        # if I_test and day >= t_test_day and not tested and (day - t_symptoms >= target_test_day):
            I_test_positive = calc_test_single(day - t_symptoms, True)
            test_results_day = day + test_delay
            test_results_day_exists = True
            # days_rel_to_sympts.append(day - t_symptoms)
            if not I_test_positive:
                # negatives[0] += 1
                t_quarantine_end = min(day + test_delay, t_quarantine_end)
            else:
                pass
                # positives[0] += 1
            tested = True
            n_tests += 1
        # code.interact(local=locals())
        if I_random_testing and not I_test_positive and np.random.binomial(n=1, p=test_freq):
        # if I_random_testing and not I_test_positive and first_test <= day < t_infectious_end and 0 <= (day - first_test) % test_interval < 1:
            I_test_positive = calc_test_single(day - t_symptoms, True)
            if I_test_positive:
                if I_isolation:
                    t_isolation = min(t_isolation, day)
                else:
                    t_isolation = day
                    I_isolation = True
                t_isolation_end = t_isolation + symp_quarantine_length
                t_quarantine_end = min(t_quarantine_end, t_isolation_end)
            tested = True
            n_tests += 1
        # infectious, but not quarantined or isolated
        day += 1
    # print("billed tracer days", billed_tracer_days, quarantine_days, isolation_days, t_isolation_end - t_isolation)
    return (quarantine_days, isolation_days, transmission_days, n_tests, monitoring_days, false_isolation_days, billed_tracer_days)

@njit
def draw_infections_from_contacts(num_individuals, g0_COVID, g0_symptoms,
                                  g0_original_case, g0_superspreader,
                                  g0_test_positive, g0_incubation_infect,
                                  g0_t_self_isolate, g0_I_contacted,
                                  g0_contacts, trace_delay=0, test_delay=0,
                                  trace=False, trace_superspreader=False,
                                  ttq=False, early_release=False, vacc_eff=0.,
                                  tfn=0.0, wait_before_testing=0, ttq_double=False,
                                  household_SAR=base_household_cal, external_SAR=base_external_cal,
                                  SS_mult=SS_mult_cal, phys_mult=phys_mult_cal,
                                  monitor=False, seed=0):
    np.random.seed(seed)
    n_age = List()
    t_exposure = List()
    t_last_exposure = List()
    original_case = List()
    infected_by = List()
    I_COVID = List()
    I_household = List()
    infected_by_presymp = List()
    infected_by_asymp = List()
    successful_traces = List()

    quarantine_days_of_uninfected = 0
    tests_of_uninfected = 0
    monitoring_days_of_uninfected = 0
    # SAR_SS_household = max(min(frac_infs_from_SS * symptoms_household_SAR / ((1 + frac_infs_from_SS) / (1 + 1 / freq_superspreaders)), 1), 0)
    # SAR_SS_ext = max(min(frac_infs_from_SS * symptoms_external_SAR / ((1 + frac_infs_from_SS) / (1 + 1 / freq_superspreaders)), 1), 0)
    # SAR_reg_household =  max(min((symptoms_household_SAR - SAR_SS_household * freq_superspreaders) / (1 - freq_superspreaders), 1), 0)
    # SAR_reg_ext =  max(min((symptoms_external_SAR - SAR_SS_ext * freq_superspreaders) / (1 - freq_superspreaders), 1), 0)

    uninfected_source = List()
    uninfected_exposure = List()
    uninfected_household = List()

    household_exposures = 0
    household_infections = 0
    external_exposures = 0
    external_infections = 0

    num_downstream_contacts_by_id = np.zeros(num_individuals)
    num_downstream_traces_by_id = np.zeros(num_individuals)
    SARS = List()

    for i in range(num_individuals):
        if not g0_COVID[i]:
            continue
        for j in range(len(g0_contacts[i])):
            num_downstream_contacts_by_id[i] += 1
            traced = np.random.binomial(n=1, p=1 - tfn) > 0.99
            if traced and g0_I_contacted[i] and trace:
                num_downstream_traces_by_id[i] += 1
            infected = False
            SAR = 1.0
            if g0_contacts[i][j][2]:
                SAR = household_SAR
                household_exposures += 1
            else:
                SAR = external_SAR
                external_exposures += 1
            if g0_contacts[i][j][7]:
                SAR = SAR * phys_mult
            if g0_superspreader[i]:
                SAR = SAR * SS_mult
            if not g0_symptoms[i]: # or g0_contacts[i][j][6] < days_infectious_before_symptoms:
                SAR = SAR * asymp_infectiousness
            if g0_symptoms[i] and g0_contacts[i][j][4] < g0_incubation_infect[i] + days_infectious_before_symptoms:
                SAR = SAR * presymp_infectiousness
            SAR = min(max(SAR, 0.), 1.)
            if i == 0:
                SARS.append(SAR)
            infected = np.random.binomial(n=1, p=SAR)
            if infected and g0_contacts[i][j][8]:
                if np.random.binomial(n=1, p=vacc_eff):
                    infected = False
            if infected:
                if g0_contacts[i][j][2]:
                    household_infections += 1
                else:
                    external_infections += 1
                if g0_contacts[i][j][4] < g0_incubation_infect[i] + days_infectious_before_symptoms and g0_symptoms[i]:
                    infected_by_presymp.append(True)
                else:
                    infected_by_presymp.append(False)
                if not g0_symptoms[i]:
                    infected_by_asymp.append(True)
                else:
                    infected_by_asymp.append(False)
                successful_traces.append(traced)
                n_age.append(g0_contacts[i][j][0])
                t_exposure.append(g0_contacts[i][j][4])
                t_last_exposure.append(g0_contacts[i][j][6])
                infected_by.append(g0_contacts[i][j][5])
                I_household.append(g0_contacts[i][j][2])
                I_COVID.append(1)
                if g0_original_case[i] == -1:
                    original_case.append(g0_contacts[i][j][5])
                else:
                    original_case.append(g0_original_case[i])
            else:
                if g0_I_contacted[i] and trace and traced:
                    # trace_start = g0_t_self_isolate[i] + trace_delay + test_delay
                    # quarantine_end = asymp_quarantine_length + g0_contacts[i][j][6]
                    uninfected_source.append(i)
                    uninfected_exposure.append(g0_contacts[i][j][6])
                    uninfected_household.append(g0_contacts[i][j][2])
    # print('SARS', SARS)
    # print(g0_contacts[0])
    return (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID, I_household, successful_traces,
            quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
            infected_by_presymp, infected_by_asymp, uninfected_source, uninfected_exposure, uninfected_household,
            num_downstream_contacts_by_id, num_downstream_traces_by_id, household_infections / household_exposures if household_exposures > 0 else 0.,
            external_infections / external_exposures if external_exposures > 0 else 0.)

@njit
def calc_pooled_tests(g1_infected_by, g1_t_incubation, g0_I_contacted,
                      g0_t_self_isolate, successful_traces, trace_delay, test_delay,
                      dropouts, test_day, seed=0):
    positive_tests = np.zeros(len(g0_t_self_isolate))
    dropout_sum = np.sum(dropouts, axis=1)
    tests_per_parent_case = np.zeros(len(g0_t_self_isolate))
    tests_per_case = np.zeros(len(g1_t_incubation))

    for i in range(len(g1_I_symptoms)):
        if not successful_traces[i] or dropout_sum[i] > 0. or not g0_I_contacted[parent_case]:
            continue
        parent_case = int(g1_infected_by[i])
        if tests_per_parent_case[i] >= pooling_max:
            continue
        time_of_symptoms_rel_to_monitoring = g1_t_incubation[i] - (g0_t_self_isolate[parent_case] + trace_delay + test_delay)
        positive_tests[parent_case] += calc_test_single(test_day - time_of_symptoms_rel_to_monitoring, True)
        tests_per_parent_case[parent_case] += 1
        if tests_per_parent_case[parent_case] < 2.:
            tests_per_case[i] += 1
    return (positive_tests, tests_per_case)

@njit
def calc_symptoms_by_day(g1_I_symptoms, g1_infected_by, g1_t_incubation, g1_false_positive, g0_I_contacted,
                         g0_t_self_isolate, successful_traces, trace_delay, test_delay,
                         uninf_g1_source, uninf_g1_false_positive,
                         dropouts, tracking_days):
    symptoms_by_day = np.zeros((len(g0_t_self_isolate), tracking_days))
    symptoms_by_day_infected = np.zeros((len(g0_t_self_isolate), tracking_days))
    symptoms_by_day_uninf = np.zeros((len(g0_t_self_isolate), tracking_days))

    time_of_symptoms_rel_to_monitoring = np.ones(len(g1_I_symptoms)) * -1
    time_of_symptoms = np.ones(len(g1_I_symptoms)) * -1
    # the infected
    # print('initial', tracking_days, symptoms_by_day, dropouts)
    for i in range(len(g1_I_symptoms)):
        if not successful_traces[i]:
            continue
        seen_symptoms = False
        dropped_out = False
        parent_case = int(g1_infected_by[i])
        if not g0_I_contacted[parent_case]:
            continue
        time_of_symptoms[i] = g1_false_positive[i]
        if g1_I_symptoms[i]:
            time_of_symptoms[i] = min(g1_false_positive[i], g1_t_incubation[i])
        time_of_symptoms_rel_to_monitoring[i] = time_of_symptoms[i] - (g0_t_self_isolate[parent_case] + trace_delay + test_delay)
        for j in range(tracking_days):
            # print("sanity", i, parent_case, j)
            if dropouts[i, j]:
                dropped_out = True
            if seen_symptoms:
                symptoms_by_day[parent_case, j] += 1
                symptoms_by_day_infected[parent_case, j] += 1
            elif not dropped_out and j + g0_t_self_isolate[parent_case] + trace_delay + test_delay >= time_of_symptoms[i]:
                seen_symptoms = True
                symptoms_by_day[parent_case, j] += 1
                symptoms_by_day_infected[parent_case, j] += 1
        # print("stats", i, parent_case, g1_I_symptoms[i], time_of_symptoms[i], g1_false_positive[i], g1_t_incubation[i], 0 + g0_t_self_isolate[parent_case] + trace_delay + test_delay, symptoms_by_day)


    # uninfected
    for i in range(len(uninf_g1_source)):
        seen_symptoms = False
        dropped_out = False
        parent_case = int(uninf_g1_source[i])
        for j in range(tracking_days):
            if dropouts[i + len(g1_I_symptoms), j]:
                dropped_out = True
            if seen_symptoms:
                symptoms_by_day[parent_case, j] += 1
                symptoms_by_day_uninf[parent_case, j] += 1
            elif not dropped_out and j + g0_t_self_isolate[parent_case] + trace_delay + test_delay >= uninf_g1_false_positive[i]:
                symptoms_by_day[parent_case, j] += 1
                symptoms_by_day_uninf[parent_case, j] += 1
                seen_symptoms = True

    # print("trace starts", g0_t_self_isolate + trace_delay + test_delay)
    # print("time of symptoms inf", time_of_symptoms_rel_to_monitoring)
    # print("time of genuine symptoms", g1_t_incubation)
    # print("infected symptoms by day", symptoms_by_day_infected)
    # # print("inf dropouts", dropouts[:len(g1_t_incubation)])
    # print("time of symptoms uninf", uninf_g1_false_positive)
    # print("uninfected symptoms by day", symptoms_by_day_uninf)
    # # print("uninf dropouts", dropouts[len(g1_t_incubation):])

    return symptoms_by_day

@njit
def calc_quarantine_uninfected(trace_start, quarantine_end, t_exposure,
                               wait_before_testing=0, test_day=0,
                               test_delay=0,
                               early_release_day=0, early_release=False,
                               n_consec_test=0, test_release=False,
                               I_monitor=False, monitor_start=0,
                               monitor_end=0, symptom_false_positive=0,
                               dropouts=np.zeros(1).reshape(1, -1), precalc_dropout=0,
                               quarantine_dropout_rate=quarantine_dropout_rate_default):
    day = trace_start
    tests = 0
    consec_negative_tests = 0
    quarantine_dropout_chance = quarantine_dropout_rate
    quarantine_dropout_rate_pos_test = 0.0 * quarantine_dropout_rate
    quarantine_dropout_rate_not_released = quarantine_dropout_rate * (1 - dropout_reduction_for_symptoms)
    quarantine_dropout_rate_neg_test = quarantine_dropout_rate * 2

    quarantine_days = 0
    monitor_days = 0
    end = quarantine_end
    test_results_day = 0
    false_positive_isolation_days = 0
    I_self_isolate = False
    t_self_isolate = 0
    t_self_isolate_end = -1

    # print("np.shape", np.shape(dropouts), np.shape(dropouts)[0], day - trace_start, day - trace_start < precalc_dropout)

    if I_monitor:
        end = max(quarantine_end, monitor_end)
    while day < quarantine_end or (I_monitor and day < monitor_end) or (I_self_isolate and day < t_self_isolate_end):
        if trace_start <= day < quarantine_end:
            quarantine_days += 1
        elif I_self_isolate and t_self_isolate <= day < t_self_isolate_end:
            false_positive_isolation_days += 1
        elif I_monitor and monitor_start <= day < monitor_end:
            monitor_days += 1
            # print("monitoring sub", day, t_exposure, symptom_false_positive, not I_self_isolate)
            if day >= t_exposure + symptom_false_positive and not I_self_isolate:
                I_self_isolate = True
                t_self_isolate = day
                t_self_isolate_end = t_self_isolate + symp_quarantine_length

        if test_release and day >= test_results_day and tests >= 1:
            if consec_negative_tests >= n_consec_test:
                quarantine_end = day
            quarantine_dropout_chance = quarantine_dropout_rate_neg_test
        if trace_start <= day < quarantine_end:
            if precalc_dropout > 0 and day - trace_start < precalc_dropout:
                if quarantine_dropout_chance == quarantine_dropout_rate and dropouts[0][int(math.floor(day - trace_start))] > 0.:
                    quarantine_end = day
                    if I_monitor:
                        monitor_end = day
            elif np.random.binomial(n=1, p=quarantine_dropout_chance):
                quarantine_end = day
                if I_monitor:
                    monitor_end = day
        if test_release and day - test_results_day >= wait_before_testing and day >= test_day and trace_start <= day < quarantine_end:
            test_results_day = day + test_delay
            tests += 1
            consec_negative_tests += 1
        if early_release and day >= early_release_day:
            quarantine_dropout_chance = quarantine_dropout_rate_not_released
        day += 1
    # print(n_consec_test, test_release, initial_quarantine_length, quarantine_days)
    # if report:
    #     print("monitoring", day, I_monitor, monitor_end, quarantine_end, trace_start, false_positive_isolation_days)
    # print("quarantine", t_exposure, trace_start, quarantine_end)
    return (quarantine_days, tests, monitor_days, false_positive_isolation_days)

@jit(nopython=True, parallel=True)
def calc_quarantine_days_of_uninfected(uninf_g1_source, uninf_g1_exposure_day, uninf_g1_household,
                                       symptom_false_positive,
                                       g1_I_symptoms, g1_t_symptoms,
                                       g0_t_self_isolate, g0_test_positive,
                                       g0_I_contacted, g0_superspreader, ttq=False,
                                       ttq_double=False, trace=False,
                                       trace_delay=0, test_delay=0, wait_before_testing=0,
                                       wait_until_testing=0,
                                       early_release=False,
                                       quarantine_by_parent_case_release=np.zeros(1),
                                       early_release_by_parent_case=np.zeros(1),
                                       monitor=False, dropouts=np.zeros(1).reshape(1, -1),
                                       precalc_dropout=0, quarantine_dropout_rate=quarantine_dropout_rate_default,
                                       hold_hh=False,
                                       seed=0):
    np.random.seed(seed)
    quarantine_days_of_uninfected = np.zeros(len(g0_t_self_isolate))
    tests_of_uninfected = np.zeros(len(g0_t_self_isolate))
    monitoring_days_of_uninfected = np.zeros(len(g0_t_self_isolate))
    isolation_days_of_uninfected = np.zeros(len(g0_t_self_isolate))
    total_contacts = 0.
    released_contacts = 0.
    isolation_events = 0.
    # print("uninfected ", quarantine_dropout_rate)
    for i in prange(len(uninf_g1_source)):
        parent_case = int(uninf_g1_source[i])
        if g0_I_contacted[parent_case] and trace:
            total_contacts += 1
            trace_start = g0_t_self_isolate[parent_case] + trace_delay + test_delay
            quarantine_end = asymp_quarantine_length + uninf_g1_exposure_day[i]
            test_day = max(uninf_g1_exposure_day[i] + wait_before_testing + wait_until_testing, trace_start + wait_before_testing)
            monitor_start = trace_start
            monitor_end = quarantine_end
            if early_release:
                if (not quarantine_by_parent_case_release[parent_case]) or (hold_hh and uninf_g1_household[i]):
                    tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                     quarantine_end=quarantine_end,
                                                     early_release=True,
                                                     early_release_day=early_release_by_parent_case[parent_case] + trace_start,
                                                     I_monitor=monitor,
                                                     monitor_start=monitor_start,
                                                     monitor_end=monitor_end,
                                                     symptom_false_positive=symptom_false_positive[i],
                                                     t_exposure=uninf_g1_exposure_day[i],
                                                     dropouts=dropouts[i].reshape(1, -1),
                                                     precalc_dropout=precalc_dropout,
                                                     quarantine_dropout_rate=quarantine_dropout_rate)
                elif ttq:
                    tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                     quarantine_end=quarantine_end,
                                                     I_monitor=monitor,
                                                     n_consec_test=1, test_release=True,
                                                     test_delay=test_delay,
                                                     test_day=test_day,
                                                     wait_before_testing=wait_before_testing,
                                                     monitor_start=monitor_start,
                                                     monitor_end=monitor_end,
                                                     symptom_false_positive=symptom_false_positive[i],
                                                     t_exposure=uninf_g1_exposure_day[i],
                                                     dropouts=dropouts[i].reshape(1, -1),
                                                     precalc_dropout=precalc_dropout,
                                                     quarantine_dropout_rate=quarantine_dropout_rate)
                    released_contacts += 1
                else:
                    tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                     quarantine_end=min(quarantine_end,
                                                                        trace_start + early_release_by_parent_case[parent_case]),
                                                     I_monitor=monitor,
                                                     monitor_start=monitor_start,
                                                     monitor_end=monitor_end,
                                                     symptom_false_positive=symptom_false_positive[i],
                                                     t_exposure=uninf_g1_exposure_day[i],
                                                     dropouts=dropouts[i].reshape(1, -1),
                                                     precalc_dropout=precalc_dropout,
                                                     quarantine_dropout_rate=quarantine_dropout_rate)
                    released_contacts += 1
            elif ttq:
                tmp = calc_quarantine_uninfected(trace_start=trace_start, quarantine_end=quarantine_end,
                                                 wait_before_testing=wait_before_testing,
                                                 test_delay=test_delay,
                                                 test_day=test_day,
                                                 n_consec_test=1, test_release=True, I_monitor=monitor,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end,
                                                 symptom_false_positive=symptom_false_positive[i],
                                                 t_exposure=uninf_g1_exposure_day[i],
                                                 quarantine_dropout_rate=quarantine_dropout_rate)
            elif ttq_double:
                tmp = calc_quarantine_uninfected(trace_start=trace_start, quarantine_end=quarantine_end,
                                                 wait_before_testing=wait_before_testing, test_delay=test_delay,
                                                 n_consec_test=2, test_release=True, I_monitor=monitor,
                                                 test_day=test_day,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end,
                                                 symptom_false_positive=symptom_false_positive[i],
                                                 t_exposure=uninf_g1_exposure_day[i],
                                                 quarantine_dropout_rate=quarantine_dropout_rate)
            else:
                tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                 quarantine_end=quarantine_end, I_monitor=monitor,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end,
                                                 symptom_false_positive=symptom_false_positive[i],
                                                 t_exposure=uninf_g1_exposure_day[i],
                                                 quarantine_dropout_rate=quarantine_dropout_rate)
            quarantine_days_of_uninfected[parent_case] += tmp[0]
            tests_of_uninfected[parent_case] += tmp[1]
            monitoring_days_of_uninfected[parent_case] += tmp[2]
            isolation_days_of_uninfected[parent_case] += tmp[3]
            if tmp[3] > 0:
                isolation_events += 1
    if early_release:
        release_pct = released_contacts / total_contacts
        print("percent of contacts released", release_pct)
    if monitor and total_contacts > 0 :
        isolation_pct = isolation_events / total_contacts
        print("isolation percentage of uninfected", isolation_pct)
    # print("isolation days of uninfected", monitor, isolation_days_of_uninfected)
    return (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected, isolation_days_of_uninfected)

def draw_traced_generation_from_contacts(g0_cases, contacts, trace=False,
                                         trace_superspreader=False, monitor_non_SS=False,
                                         bidirectional_SS=False,
                                         trace_delay=0, test_delay=0, trace_false_negative=0.0,
                                         ttq=False, early_release=False,
                                         early_release_day=0, early_release_threshold=0.0,
                                         wait_before_testing=0, wait_until_testing=0, vacc_eff=0.,
                                         ttq_double=False,
                                         household_SAR=base_household_cal, external_SAR=base_external_cal,
                                         SS_mult=SS_mult_cal, phys_mult=phys_mult_cal,
                                         monitor=False, small_group_delay=False, small_group_threshold=0,
                                         small_group_extra=0, cluster_pooling_release=False, seed=0,
                                         quarantine_dropout_rate=quarantine_dropout_rate_default,
                                         hold_hh=True):
    # print("draw traced ", quarantine_dropout_rate)
    (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID, I_household, successful_traces,
     quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
     infected_by_presymp, infected_by_asymp, uninfected_source, uninfected_exposure, uninfected_household,
     num_downstream_contacts_by_id, num_downstream_traces_by_id, household_SAR, external_SAR) = draw_infections_from_contacts(
         num_individuals=len(g0_cases['I_COVID']),
         g0_COVID=g0_cases['I_COVID'],
         g0_symptoms=g0_cases['I_symptoms'],
         g0_original_case=g0_cases['id_original_case'],
         g0_superspreader=g0_cases['I_superspreader'],
         g0_incubation_infect=g0_cases['t_incubation_infect'],
         g0_t_self_isolate=g0_cases['t_self_isolate'],
         g0_I_contacted=g0_cases['I_contacted'],
         g0_test_positive=g0_cases['I_test_positive'],
         g0_contacts=contacts,
         trace_delay=trace_delay,
         test_delay=test_delay,
         vacc_eff=vacc_eff,
         trace=trace,
         ttq=ttq,
         ttq_double=ttq_double,
         early_release=early_release,
         tfn=trace_false_negative,
         wait_before_testing=wait_before_testing,
         household_SAR=household_SAR, external_SAR=external_SAR,
         SS_mult=SS_mult, phys_mult=phys_mult,
         monitor=monitor, seed=seed)

    t_exposure = np.array(t_exposure)
    num_g1_cases = len(n_age)
    g1_cases = draw_seed_index_cases(num_individuals=num_g1_cases,
                                     age_vector=age_vector_US,
                                     t_exposure=t_exposure,
                                     skip_days=True,
                                     seed=seed + 1)
    g1_cases['t_last_exposure'] = np.array(t_last_exposure)
    g1_cases['n_age'] = np.array(n_age)
    g1_cases['id_original_case'] = np.array(original_case)
    g1_cases['id_infected_by'] = np.array(infected_by)
    g1_cases['I_COVID'] = np.array(I_COVID)
    g1_cases['I_infected_by_asymp'] = np.array(infected_by_asymp)
    g1_cases['I_infected_by_presymp'] = np.array(infected_by_presymp)
    g1_cases['household_SAR'] = household_SAR
    g1_cases['external_SAR'] = external_SAR
    g1_cases['I_tracing_failed'] = 1 - np.array(successful_traces)
    g1_cases['I_household'] = np.array(I_household)
    uninfected_source = np.array(uninfected_source)
    uninfected_exposure = np.array(uninfected_exposure)

    quarantine_by_parent_case_release = np.zeros(len(g0_cases['I_COVID']))

    uninf_false_positive = np.random.geometric(p=symptom_false_positive_chance, size=len(uninfected_source)) + uninfected_exposure
    if early_release_day is None:
        early_release_day = 0

    precalc_dropout = 0
    dropouts = np.zeros(len(g1_cases['I_symptoms']) + len(uninfected_source)).reshape(-1, 1)
    # symptom check is one day before release
    quarantine_early_release_day = early_release_day + 1
    early_release_by_parent_case = np.ones(len(g0_cases['I_COVID'])) * quarantine_early_release_day
    if early_release:
        dropouts = np.random.binomial(n=1, p=quarantine_dropout_rate, size=(len(g1_cases['I_symptoms']) + len(uninfected_source), early_release_day + small_group_extra + 1))
        symptoms_by_day = calc_symptoms_by_day(
            g1_I_symptoms=g1_cases['I_symptoms'],
            g1_infected_by=g1_cases['id_infected_by'],
            g1_t_incubation=g1_cases['t_incubation'],
            g1_false_positive=g1_cases['t_false_positive'],
            g0_I_contacted=g0_cases['I_contacted'],
            g0_t_self_isolate=g0_cases['t_self_isolate'],
            successful_traces=successful_traces,
            trace_delay=trace_delay, test_delay=test_delay,
            uninf_g1_source=uninfected_source,
            uninf_g1_false_positive=uninf_false_positive,
            dropouts=dropouts,
            tracking_days=early_release_day + small_group_extra + 1)
        percentage_by_day = np.nan_to_num(symptoms_by_day / (num_downstream_traces_by_id.reshape(-1, 1)))
        quarantine_by_parent_case_release = (percentage_by_day[:, early_release_day] < early_release_threshold)
        # print("early release stats", early_release_threshold, symptoms_by_day, num_downstream_traces_by_id, percentage_by_day, quarantine_by_parent_case_release)
        # print("symptoms by day", symptoms_by_day)
        # print("percentage by day", percentage_by_day)
        # print("downstream contacts", num_downstream_contacts_by_id)
        # print("downstream traces", num_downstream_traces_by_id)
        # print("superspreader", g0_cases['I_superspreader'])
        if small_group_delay:
            for i, _ in enumerate(quarantine_by_parent_case_release):
                if num_downstream_contacts_by_id[i] <= small_group_threshold:
                    quarantine_by_parent_case_release[i] = percentage_by_day[i, early_release_day + small_group_extra] < early_release_threshold
                    early_release_by_parent_case[i] = quarantine_early_release_day + small_group_extra
        precalc_dropout = early_release_day + small_group_extra + 1
        (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected, isolation_days_of_uninfected) = calc_quarantine_days_of_uninfected(
            uninf_g1_source=uninfected_source,
            uninf_g1_exposure_day=uninfected_exposure,
            uninf_g1_household=uninfected_household,
            g1_I_symptoms=g1_cases['I_symptoms'],
            g1_t_symptoms=g1_cases['t_incubation'],
            g0_t_self_isolate=g0_cases['t_self_isolate'],
            g0_test_positive=g0_cases['I_test_positive'],
            g0_I_contacted=g0_cases['I_contacted'],
            g0_superspreader=g0_cases['I_superspreader'],
            ttq=ttq,
            ttq_double=ttq_double,
            trace=trace,
            early_release=True,
            trace_delay=trace_delay,
            test_delay=test_delay,
            wait_before_testing=wait_before_testing,
            wait_until_testing=wait_until_testing,
            quarantine_by_parent_case_release=quarantine_by_parent_case_release,
            early_release_by_parent_case=early_release_by_parent_case,
            monitor=monitor,
            symptom_false_positive=uninf_false_positive, dropouts=(dropouts[len(g1_cases['I_symptoms']):, :]).reshape(len(uninfected_source), early_release_day + small_group_extra + 1),
            precalc_dropout=precalc_dropout,
            seed=seed,
            quarantine_dropout_rate=quarantine_dropout_rate,
            hold_hh=hold_hh)
    else:
        (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected, isolation_days_of_uninfected) = calc_quarantine_days_of_uninfected(
            uninf_g1_source=uninfected_source,
            uninf_g1_exposure_day=uninfected_exposure,
            uninf_g1_household=uninfected_household,
            g1_I_symptoms=g1_cases['I_symptoms'],
            g1_t_symptoms=g1_cases['t_incubation'],
            g0_t_self_isolate=g0_cases['t_self_isolate'],
            g0_test_positive=g0_cases['I_test_positive'],
            g0_I_contacted=g0_cases['I_contacted'],
            g0_superspreader=g0_cases['I_superspreader'],
            ttq=ttq,
            ttq_double=ttq_double,
            trace=trace,
            wait_before_testing=wait_before_testing,
            wait_until_testing=wait_until_testing,
            trace_delay=trace_delay,
            test_delay=test_delay,
            monitor=monitor,
            symptom_false_positive=uninf_false_positive,
            seed=seed,
            quarantine_dropout_rate=quarantine_dropout_rate,
            hold_hh=hold_hh)

    (g1_cases['n_quarantine_days'], g1_cases['n_transmission_days'], g1_cases['n_isolation_days'],
     secondary_cases_traced, secondary_cases_monitored, g1_cases['n_monitoring_days'],
     g1_cases['n_tests'], g1_cases['n_false_positive_isolation_days'],
     g1_cases['n_billed_tracer_days']) = fill_QII(
        t_exposure=t_exposure, t_last_exposure=g1_cases['t_last_exposure'],
        trace_delay=trace_delay, test_delay=test_delay,
        t_self_isolate=g1_cases['t_self_isolate'],
        I_self_isolate=g1_cases['I_self_isolate'],
        t_incubation_infect=g1_cases['t_incubation_infect'],
        t_false_positive=g1_cases['t_false_positive'],
        t_incubation=g1_cases['t_incubation'],
        I_symptoms=g1_cases['I_symptoms'],
        I_household=g1_cases['I_household'],
        id_infected_by=g1_cases['id_infected_by'],
        num_g1_cases=num_g1_cases,
        g0_I_contacted=g0_cases['I_contacted'],
        g0_I_self_isolate=g0_cases['I_self_isolate'],
        g0_t_self_isolate=g0_cases['t_self_isolate'],
        g0_I_test_positive=g0_cases['I_test_positive'],
        g0_I_symptoms=g0_cases['I_symptoms'],
        g0_I_superspreader=g0_cases['I_superspreader'],
        trace=trace,
        ttq=ttq,
        ttq_double=ttq_double,
        early_release=early_release,
        quarantine_by_parent_case_release=quarantine_by_parent_case_release,
        early_release_by_parent_case=early_release_by_parent_case,
        wait_before_testing=wait_before_testing,
        wait_until_testing=wait_until_testing,
        monitor=monitor,
        precalc_dropout=precalc_dropout,
        dropouts=dropouts[:len(g1_cases['I_symptoms']), :].reshape(len(g1_cases['I_symptoms']), early_release_day + small_group_extra + 1),
        false_negative_traces=g1_cases['I_tracing_failed'],
        seed=seed,
        quarantine_dropout_rate=quarantine_dropout_rate,
        hold_hh=hold_hh)
    print("%i secondary cases traced" % secondary_cases_traced)
    print("%i secondary cases monitored" % secondary_cases_monitored)
    g1_cases['n_quarantine_days_of_uninfected'] = quarantine_days_of_uninfected
    g1_cases['n_tests_of_uninfected'] = tests_of_uninfected
    g1_cases['n_monitoring_days_of_uninfected'] = monitoring_days_of_uninfected
    g1_cases['secondary_cases_traced'] = secondary_cases_traced
    g1_cases['secondary_cases_monitored'] = secondary_cases_monitored
    g1_cases['n_isolation_days_of_uninfected'] = isolation_days_of_uninfected
    return (g1_cases, uninfected_source, uninfected_exposure)

@jit(nopython=True, parallel=True)
def get_transmission_days(t_exposure, t_last_exposure,
                          I_isolation, t_isolation,
                          t_infectious, t_false_positive,
                          I_symptoms, t_symptoms,
                          test_freq=0,
                          test_delay=0.,
                          I_random_testing=False):
    n_quarantine_days = np.zeros(len(t_exposure))
    n_transmission_days = np.zeros(len(t_exposure))
    n_isolation_days = np.zeros(len(t_exposure))
    n_tests = np.zeros(len(t_exposure))
    n_monitoring_days = np.zeros(len(t_exposure))
    n_false_positive_isolation_days = np.zeros(len(t_exposure))
    n_billed_tracer_days = np.zeros(len(t_exposure))

    for i in prange(len(t_exposure)):
        (n_quarantine_days[i], n_isolation_days[i], n_transmission_days[i], n_tests[i], n_monitoring_days[i], n_false_positive_isolation_days[i], n_billed_tracer_days[i]) = calculate_QII_days(t_exposure=t_exposure[i], t_last_exposure=t_last_exposure[i],
                                                                                                 I_isolation=I_isolation[i], t_isolation=t_isolation[i],
                                                                                                 t_infectious=t_infectious[i],
                                                                                                 I_symptoms=I_symptoms[i], t_symptoms=t_symptoms[i],
                                                                                                 test_freq=test_freq, test_delay=test_delay,
                                                                                                 I_random_testing=I_random_testing,
                                                                                                 t_false_positive=t_false_positive[i])
    return (n_quarantine_days, n_transmission_days, n_isolation_days, n_tests,
            n_monitoring_days, n_false_positive_isolation_days, n_billed_tracer_days)

@jit(nopython=True,parallel=True)
def aggregate_infections(case_ids, n_cases_g0, n_cases_g1):
    aggregated_infections = np.zeros(n_cases_g0)
    for i in prange(n_cases_g1):
        aggregated_infections[int(case_ids[i])] += 1
    return aggregated_infections

def aggregate_stats_per_index_case(g0_cases, g1_cases, g2_cases, g0_contacts):
    num_tests = np.copy(g1_cases['n_tests_of_uninfected'])
    num_quarantine_days = np.copy(g1_cases['n_quarantine_days_of_uninfected'])
    false_positive_days = np.copy(g1_cases['n_isolation_days_of_uninfected'])
    num_cases_g1 = np.zeros(len(g0_cases['I_COVID']))
    num_cases_g2 = np.zeros(len(g0_cases['I_COVID']))
    billed_tracer_days = np.zeros(len(g0_cases['I_COVID']))
    num_contacts = np.zeros(len(g0_cases['I_COVID']))
    num_contacts_household = np.zeros(len(g0_cases['I_COVID']))
    num_contacts_physical = np.zeros(len(g0_cases['I_COVID']))
    num_deaths = np.zeros(len(g0_cases['I_COVID']))

    for (case, _) in enumerate(g0_cases['I_COVID']):
        num_contacts[case] += len(g0_contacts[case])
        for contact in g0_contacts[case]:
            if contact[2]:
                num_contacts_household[case] += 1
            if contact[7]:
                num_contacts_physical[case] += 1
    for (case, _) in enumerate(g1_cases['I_COVID']):
        if g1_cases['I_COVID'][case]:
            original_infector = int(g1_cases['id_original_case'][case])
            num_tests[original_infector] += g1_cases['n_tests'][case]
            num_quarantine_days[original_infector] += g1_cases['n_quarantine_days'][case]
            false_positive_days[original_infector] += g1_cases['n_false_positive_isolation_days'][case]
            num_cases_g1[original_infector] += 1
            billed_tracer_days[original_infector] += g1_cases['n_billed_tracer_days'][case]
            num_deaths[original_infector] += age_specific_IFRs[int(g1_cases['n_age'][case])]
    for (case, _) in enumerate(g2_cases['I_COVID']):
        if g2_cases['I_COVID'][case]:
            original_infector = int(g2_cases['id_original_case'][case])
            num_cases_g2[original_infector] += 1
            num_deaths[original_infector] += age_specific_IFRs[int(g2_cases['n_age'][case])]
    return (num_tests, num_quarantine_days, false_positive_days, billed_tracer_days, num_cases_g1, num_cases_g2, num_contacts, num_contacts_household, num_contacts_physical, num_deaths)

""" contact tracing, quarantine all contacts immediately on index case self-isolation """
def contact_tracing(num_index_cases, trace=False,
                    trace_delay=0,
                    test_delay=0,
                    trace_false_negative=0.0,
                    cases_contacted=1.0,
                    ttq=False,
                    base_reduction=0.0,
                    early_release=False,
                    early_release_day=0,
                    early_release_threshold=1.e-6,
                    small_group_delay=False,
                    small_group_threshold=0,
                    small_group_extra=0,
                    wait_before_testing=0,
                    wait_until_testing=0,
                    ttq_double=False,
                    cluster_pooling_release=False,
                    monitor=False,
                    hold_hh=False,
                    seed=0,
                    quarantine_dropout_rate=quarantine_dropout_rate_default,
                    output_json=False):
    np.random.seed(seed)
    g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US, cases_contacted, seed=seed, initial=True)
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    n_contacted = np.sum(g0_cases['I_contacted'])
    g0_contacts = draw_contact_generation(g0_cases, base_reduction=base_reduction, seed=seed)
    # print("g0", g0_cases)
    # print("g0_contacts", g0_contacts)
    print("positive tests: %i" % np.sum(g0_cases['I_test_positive']))
    print('num exposures: %i household, %i external' % (np.sum([int(np.sum([y[2] for y in x])) for x in list(g0_contacts)]),
                                                        np.sum([int(np.sum([not y[2] for y in x])) for x in list(g0_contacts)])))
    traces_per_positive = np.sum([len(g0_contacts[i]) for i in np.where(g0_cases['I_contacted'])[0]]) / n_contacted
    print("avg traces per positive: %f" % traces_per_positive)

    print("quarantine dropout rate ", quarantine_dropout_rate)
    print("quarantine dropout rate observed symptoms ", quarantine_dropout_rate * (1 - dropout_reduction_for_symptoms))
    (g1_cases, g1_uninf_source, g1_uninf_t_exposure) = draw_traced_generation_from_contacts(g0_cases=g0_cases, contacts=g0_contacts,
                                                    trace=trace,
                                                    trace_delay=trace_delay, test_delay=test_delay,
                                                    trace_false_negative=trace_false_negative,
                                                    ttq=ttq, early_release=early_release, early_release_day=early_release_day,
                                                    early_release_threshold=early_release_threshold,
                                                    wait_before_testing=wait_before_testing,
                                                    wait_until_testing=wait_until_testing,
                                                    ttq_double=ttq_double,
                                                    monitor=monitor,
                                                    seed=seed,
                                                    small_group_delay=small_group_delay,
                                                    small_group_threshold=small_group_threshold,
                                                    small_group_extra=small_group_extra,
                                                    cluster_pooling_release=cluster_pooling_release,
                                                    quarantine_dropout_rate=quarantine_dropout_rate,
                                                    hold_hh=hold_hh)
    num_g0_exposures = np.sum([int(np.sum([y[2] for y in x])) for x in list(g0_contacts)]) + np.sum([int(np.sum([not y[2] for y in x])) for x in list(g0_contacts)])
    num_g1_cases = np.sum(g1_cases['I_COVID'])
    print('new index cases: ' + str(num_g1_cases))
    g1_contacts = draw_contact_generation(g1_cases, base_reduction=base_reduction, seed=seed + 1)
    # print("g1", g1_cases)
    # print("g1_contacts", g1_contacts)
    (g2_cases, g2_uninf_source, g2_uninf_t_exposure) = draw_traced_generation_from_contacts(g0_cases=g1_cases,
                                                    contacts=g1_contacts,
                                                    trace=False,
                                                    seed=seed + 1)
    # print("g2", g2_cases)
    aggregated_infections = np.zeros(num_index_cases)
    num_g2_cases = np.sum(g2_cases['I_COVID'])
    for i in range(int(num_g2_cases)):
        aggregated_infections[int(g2_cases['id_original_case'][i])] += 1
    downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
    r0 = np.mean(downstream_COVID)
    # nbinom_fit = fit_dist.fit_neg_binom(downstream_COVID)
    # k = nbinom_fit[1][1]
    quarantine_days = list(g1_cases['n_quarantine_days'])
    downstream_asymptoms = aggregated_infections[(g0_cases['I_asymptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_symptoms = aggregated_infections[(g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    # plt.hist(downstream_COVID, bins=np.arange(0, downstream_COVID.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'Number of tertiary infections ($R_O^2$) (mean: %f, dispersion: %f)' % (r0, k))
    # plt.ylabel(r'Density')
    # plt.title(r'Idealized contact tracing')
    # plt.show()
    # np.sum(downstream_asymptoms) / np.sum(g0_cases['I_asymptoms']), np.sum(downstream_symptoms) / np.sum(g0_cases['I_symptoms']
    # print(positives)
    # print(negatives)

    n_total_tests = np.sum(g1_cases['n_tests']) + np.sum(g1_cases['n_tests_of_uninfected'])
    n_total_false_positive_isolation_days = np.sum(g1_cases['n_isolation_days_of_uninfected']) + np.sum(g1_cases['n_false_positive_isolation_days'])
    n_total_tracing_hours = n_contacted * 0.5 + 0.25 * traces_per_positive * n_contacted + (
        np.sum(g1_cases['n_monitoring_days_of_uninfected']) + np.sum(g1_cases['n_billed_tracer_days']) +
        np.sum(g1_cases['n_quarantine_days_of_uninfected']) +
        np.sum(g1_cases['n_isolation_days_of_uninfected'])) * 1 / 6.
    contact_tracer_rate = 20
    test_cost = 200
    print("tests of infected ", np.sum(g1_cases['n_tests']))
    print("tests of uninfected ", np.sum(g1_cases['n_tests_of_uninfected']))
    print("monitoring days of infected ", np.sum(g1_cases['n_monitoring_days']))
    print("monitoring days of uninfected", np.sum(g1_cases['n_monitoring_days_of_uninfected']))
    print("quarantine days of infected", np.sum(g1_cases['n_quarantine_days']))
    print("quarantine days of uninfected", np.sum(g1_cases['n_quarantine_days_of_uninfected']))
    # print("false positive isolation days of infected", np.sum(g1_cases['n_false_positive_isolation_days']))
    # print("false positive isolation days of uninfected", np.sum(g1_cases['n_isolation_days_of_uninfected']))
    print("billed tracer days of infected", np.sum(g1_cases['n_billed_tracer_days']))
    print("billed tracer days of uninfected", np.sum(g1_cases['n_monitoring_days_of_uninfected'] + g1_cases['n_quarantine_days_of_uninfected'] + g1_cases['n_isolation_days_of_uninfected']))
    print("last gen", num_g2_cases / num_g1_cases)
    print("number of transmission days", np.sum(g1_cases['n_transmission_days']) / num_g1_cases)
    print("number of quarantine days", np.sum(g1_cases['n_quarantine_days']) / num_g1_cases)

    if output_json:
        json_files = {}
        for (case, _) in enumerate(g0_cases['I_COVID']):
            json_tree = {}
            json_tree['t_exposure'] = float(g0_cases['t_exposure'][case])
            json_tree['t_last_exposure'] = float(g0_cases['t_last_exposure'][case])
            json_tree['t_incubation'] = float(g0_cases['t_incubation'][case])
            json_tree['t_incubation_infect'] = float(g0_cases['t_incubation_infect'][case])
            json_tree['I_symptoms'] = bool(g0_cases['I_symptoms'][case])
            json_tree['I_COVID'] = bool(g0_cases['I_COVID'][case])
            json_tree['I_self_isolate'] = bool(g0_cases['I_self_isolate'][case])
            json_tree['t_self_isolate_start'] = float(g0_cases['t_self_isolate'][case])
            json_tree['t_self_isolate_end'] = float(g0_cases['t_self_isolate'][case] + symp_quarantine_length)
            json_tree['t_trace_start'] = float(g0_cases['t_self_isolate'][case] + test_delay)
            json_tree['t_trace_finish'] = float(g0_cases['t_self_isolate'][case] + test_delay + trace_delay)
            json_tree['t_test_day'] = float(g0_cases['t_self_isolate'][case])
            json_tree['t_test_results_day'] = float(g0_cases['t_self_isolate'][case] + test_delay)
            json_tree['I_test_positive'] = bool(True)
            json_tree['I_contacted'] = bool(g0_cases['I_contacted'][case])
            json_tree['n_age'] = int(g0_cases['n_age'][case])
            json_tree['contacts'] = {}
            json_files[case] = json_tree
        for (case, _) in enumerate(g1_uninf_source):
            json_tree = {}
            json_tree['t_exposure'] = float(g1_uninf_exposure)
            json_tree['I_COVID'] = False

        for (case, _) in enumerate(g1_cases['I_COVID']):
            json_tree = {}
            json_tree['t_exposure'] = float(g1_cases['t_exposure'][case])
            json_tree['t_last_exposure'] = float(g1_cases['t_last_exposure'][case])
            json_tree['t_incubation'] = float(g1_cases['t_incubation'][case])
            json_tree['t_incubation_infect'] = float(g1_cases['t_incubation_infect'][case])
            json_tree['I_symptoms'] = bool(g1_cases['I_symptoms'][case])
            json_tree['I_COVID'] = bool(g1_cases['I_COVID'][case])
            json_tree['I_self_isolate'] = bool(g1_cases['I_self_isolate'][case])
            json_tree['t_self_isolate_start'] = float(g1_cases['t_self_isolate'][case])
            json_tree['t_self_isolate_end'] = float(g1_cases['t_self_isolate'][case] + symp_quarantine_length)
            json_tree['I_contacted'] = bool(1 - g1_cases['I_tracing_failed'][case])
            json_tree['contacts'] = {}
            for (case2, _) in enumerate(g2_cases['I_COVID']):
                if g2_cases['id_infected_by'][case2] == case:
                    json_tree2 = {}
                    json_tree2['t_exposure'] = float(g2_cases['t_exposure'][case])
                    json_tree2['t_last_exposure'] = float(g2_cases['t_last_exposure'][case])
                    json_tree2['t_incubation'] = float(g2_cases['t_incubation'][case])
                    json_tree2['t_incubation_infect'] = float(g2_cases['t_incubation_infect'][case])
                    json_tree2['I_symptoms'] = bool(g2_cases['I_symptoms'][case])
                    json_tree2['I_COVID'] = bool(g2_cases['I_COVID'][case])
                    json_tree2['I_self_isolate'] = float(g2_cases['I_self_isolate'][case])
                    json_tree2['t_self_isolate_start'] = float(g2_cases['t_self_isolate'][case])
                    json_tree2['t_self_isolate_end'] = float(g2_cases['t_self_isolate'][case] + symp_quarantine_length)
                    json_tree['contacts'][case2] = json_tree2
            for (case2, _) in g2_uninf_source:
                if g2_uninf_source[case2] == case:
                    json_tree2 = {}
                    json_tree2['t_exposure'] = float(g2_cases['t_exposure'][case])
                    json_tree2['t_last_exposure'] = float(g2_cases['t_last_exposure'][case])
                    json_tree2['t_incubation'] = float(g2_cases['t_incubation'][case])
                    json_tree2['t_incubation_infect'] = float(g2_cases['t_incubation_infect'][case])
                    json_tree2['I_symptoms'] = bool(g2_cases['I_symptoms'][case])
                    json_tree2['I_COVID'] = bool(g2_cases['I_COVID'][case])
                    json_tree2['I_self_isolate'] = float(g2_cases['I_self_isolate'][case])
                    json_tree2['t_self_isolate_start'] = float(g2_cases['t_self_isolate'][case])
                    json_tree2['t_self_isolate_end'] = float(g2_cases['t_self_isolate'][case] + symp_quarantine_length)
                    json_tree['contacts'][case2] = json_tree2
            json_files[g1_cases['id_infected_by'][case]]['contacts'][case] = json_tree
        for (case, _) in enumerate(g0_cases['I_COVID']):
            with open(str(case) + '.json', 'w') as f:
                json.dump(json_files[case], f)
    code.interact(local=locals())
    (g0_agg_test, g0_agg_quar, g0_agg_fp, g0_agg_bill, g0_agg_g1, g0_agg_g2,
     g0_n_contacts, g0_n_household, g0_n_physical, g0_n_deaths) = aggregate_stats_per_index_case(g0_cases, g1_cases, g2_cases, g0_contacts)
    g0_agg_hours = 0.5 + g0_n_contacts * 0.25 + (g1_cases['n_monitoring_days_of_uninfected'] + g1_cases['n_quarantine_days_of_uninfected'] + g1_cases['n_isolation_days_of_uninfected'] + g0_agg_bill) * 1/12.
    g0_agg_cost = g0_agg_test * test_cost + g0_agg_hours * contact_tracer_rate

    # g0_n_contacts_sorted = np.flip(np.sort(g0_n_contacts))
    # top_20 = np.cumsum(g0_n_contacts_sorted)[int(len(g0_n_contacts_sorted) * 0.2)]
    # SS_share_contacts = top_20 / np.sum(g0_n_contacts_sorted)
    # print("SS share contacts", SS_share_contacts)
    #
    # g0_n_household_sorted = np.flip(np.sort(g0_n_household))
    # top_20_h = np.cumsum(g0_n_household_sorted)[int(len(g0_n_household_sorted) * 0.2)]
    # SS_share_h = top_20_h / np.sum(g0_n_household_sorted)
    # print("SS share h", SS_share_h)

    # print("frac inf by asymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_asymp'])) / num_g1_cases))
    # print("frac inf by presymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_presymp'])) / num_g1_cases))

    combined = np.concatenate((g0_agg_g1.reshape(1, -1), g0_n_contacts.reshape(1, -1), g0_n_household.reshape(1, -1), g0_n_physical.reshape(1, -1), g0_cases['I_superspreader'].reshape(1, -1)), axis=0)
    combined = combined[:, np.argsort(combined[1])]

    return ((np.mean(g0_agg_g2), scipy.stats.sem(g0_agg_g2), np.std(g0_agg_g2)),
            (np.mean(g0_n_contacts), scipy.stats.sem(g0_n_contacts), np.std(g0_n_contacts)),
            (np.mean(g0_agg_quar), scipy.stats.sem(g0_agg_quar), np.std(g0_agg_quar)),
            (np.mean(g0_agg_fp), scipy.stats.sem(g0_agg_fp), np.std(g0_agg_fp)),
            (np.mean(g0_agg_test), scipy.stats.sem(g0_agg_test), np.std(g0_agg_test)),
            (np.mean(g0_agg_cost), scipy.stats.sem(g0_agg_cost), np.std(g0_agg_cost)),
            (np.mean(g0_n_deaths * 1000), scipy.stats.sem(g0_n_deaths * 1000), np.std(g0_n_deaths * 1000)),
            num_g2_cases / num_g1_cases)

    # print("\nEarly Release Monitor")
    # print(contact_tracing(num_index_cases, trace=True,
    #                       trace_delay=trace_delay,
    #                       test_delay=test_delay,
    #                       trace_false_negative=trace_false_negative,
    #                       cases_contacted=cases_contacted,
    #                       ttq=False,
    #                       base_reduction=base_reduction,
    #                       early_release=True,
    #                       early_release_day=0,
    #                       early_release_threshold=1.e-6,
    #                       wait_before_testing=0,
    #                       ttq_double=False,
    #                       monitor=True,
    #                       seed=seed))

# def main():
#     run_suite(500000, trace_delay=1, test_delay=1,
#               trace_false_negative=0.2, cases_contacted=0.8)
#     run_suite(500000, trace_delay=1, test_delay=2,
#               trace_false_negative=0.2, cases_contacted=0.8)
#     run_suite(500000, trace_delay=1, test_delay=1,
#               trace_false_negative=0.2, cases_contacted=0.8,
#               base_reduction=0.5)

def draw_samples(num_index_cases, trace_delay, test_delay, wait_before_testing,
                 trace_false_negative, cases_contacted, base_reduction=0.5, seed=0,
                 quarantine_dropout_rate=quarantine_dropout_rate_default):
    g0_cases = draw_seed_index_cases(num_individuals=num_index_cases, cases_contacted=cases_contacted,
                                     age_vector=age_vector_US, initial=True)
    g0_contacts = draw_contact_generation(g0_cases, base_reduction=base_reduction, seed=seed)
    (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID, I_household, successful_traces,
    quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
    infected_by_presymp, infected_by_asymp, uninfected_source, uninfected_exposure, uninfected_household,
    num_downstream_contacts_by_id, num_downstream_traces_by_id, household_SAR, external_SAR) = draw_infections_from_contacts(
        num_individuals=len(g0_cases['I_COVID']),
        g0_COVID=g0_cases['I_COVID'],
        g0_symptoms=g0_cases['I_symptoms'],
        g0_original_case=g0_cases['id_original_case'],
        g0_superspreader=g0_cases['I_superspreader'],
        g0_incubation_infect=g0_cases['t_incubation_infect'],
        g0_t_self_isolate=g0_cases['t_self_isolate'],
        g0_I_contacted=g0_cases['I_contacted'],
        g0_test_positive=g0_cases['I_test_positive'],
        g0_contacts=g0_contacts,
        trace_delay=trace_delay,
        early_release=True,
        trace=True,
        tfn=trace_false_negative,
        wait_before_testing=wait_before_testing,
        seed=seed,
        quarantine_dropout_rate=quarantine_dropout_rate)

    t_exposure = np.array(t_exposure)
    num_g1_cases = len(n_age)
    g1_cases = draw_seed_index_cases(num_individuals=num_g1_cases,
                                     age_vector=age_vector_US,
                                     t_exposure=t_exposure,
                                     skip_days=True,
                                     seed=seed + 1)
    g1_cases['t_last_exposure'] = np.array(t_last_exposure)
    g1_cases['n_age'] = np.array(n_age)
    g1_cases['id_original_case'] = np.array(original_case)
    g1_cases['id_infected_by'] = np.array(infected_by)
    g1_cases['I_COVID'] = np.array(I_COVID)
    g1_cases['I_infected_by_asymp'] = np.array(infected_by_asymp)
    g1_cases['I_infected_by_presymp'] = np.array(infected_by_presymp)
    g1_cases['household_SAR'] = household_SAR
    g1_cases['external_SAR'] = external_SAR
    g1_cases['false_negative_traces'] = 1 - np.array(successful_traces)
    uninfected_source = np.array(uninfected_source)
    uninfected_exposure = np.array(uninfected_exposure)

    uninf_false_positive = np.random.geometric(p=symptom_false_positive_chance, size=len(uninfected_source)) + uninfected_exposure

    dropouts = np.zeros(len(g1_cases['I_symptoms']) + len(uninfected_source)).reshape(-1, 1)
    # symptom check is one day before release

    min_size = 1
    max_size = 20
    min_day = 0
    max_day = 14

    dropouts = np.random.binomial(n=1, p=quarantine_dropout_rate, size=(len(g1_cases['I_symptoms']) + len(uninfected_source), max_day))
    symptoms_by_day = calc_symptoms_by_day(
        g1_I_symptoms=g1_cases['I_symptoms'],
        g1_infected_by=g1_cases['id_infected_by'],
        g1_t_incubation=g1_cases['t_incubation'],
        g1_false_positive=g1_cases['t_false_positive'],
        g0_I_contacted=g0_cases['I_contacted'],
        g0_t_self_isolate=g0_cases['t_self_isolate'],
        successful_traces=successful_traces,
        trace_delay=trace_delay, test_delay=test_delay,
        uninf_g1_source=uninfected_source,
        uninf_g1_false_positive=uninf_false_positive,
        dropouts=dropouts,
        tracking_days=max_day)
    percentage_by_day = np.nan_to_num(symptoms_by_day / (num_downstream_traces_by_id.reshape(-1, 1)))
    # print("percentage_by_day", percentage_by_day)
    quarantine_by_parent_case_release = (percentage_by_day < 1.e-6)

    traced_COVID_by_ID = np.zeros(len(num_downstream_traces_by_id))
    for i in range(len(g1_cases['I_symptoms'])):
        if successful_traces[i]:
            parent_case = int(g1_cases['id_infected_by'][i])
            traced_COVID_by_ID[parent_case] += 1

    # print("traced COVID by ID", traced_COVID_by_ID)
    # print("downstream traces by ID", num_downstream_traces_by_id)
    # print("downstream contacts by ID", num_downstream_contacts_by_id)
    # print("successful traces", successful_traces)

    clusters_released_by_size = np.zeros((max_size - min_size + 1, max_day - min_day + 1))
    clusters_total_by_size = np.zeros(max_size - min_size + 1)
    COVID_released_by_size = np.zeros((max_size - min_size + 1, max_day - min_day + 1))
    COVID_total_by_size = np.zeros(max_size - min_size + 1)
    total_released_by_size = np.zeros((max_size - min_size + 1, max_day - min_day + 1))

    for i in range(len(num_downstream_traces_by_id)):
        size = int(num_downstream_traces_by_id[i])
        if min_size <= size < max_size:
            clusters_total_by_size[size] += 1
            COVID_total_by_size[size] += traced_COVID_by_ID[i]
        else:
            continue
        for day in range(max_day):
            if percentage_by_day[i][day] < 1.e-6:
                clusters_released_by_size[size][day] += 1
                COVID_released_by_size[size][day] += traced_COVID_by_ID[i]
                total_released_by_size[size][day] += size

    print("clusters_total_by_size", clusters_total_by_size)
    print("COVID total by size", COVID_total_by_size)
    #
    # print("clusters released by size", clusters_released_by_size)
    # print("COVID released by size", COVID_released_by_size)

    print("fraction released by size", clusters_released_by_size / clusters_total_by_size.reshape(-1, 1))
    print("fraction COVID released by size", COVID_released_by_size / COVID_total_by_size.reshape(-1, 1))
    print("posterior prob of infetion", COVID_released_by_size / total_released_by_size)
    posterior_prob = COVID_released_by_size / total_released_by_size
    plt.plot(np.array(range(14)), posterior_prob[1][:14], label="Size 1")
    plt.plot(np.array(range(14)), posterior_prob[3][:14], label="Size 3")
    plt.plot(np.array(range(14)), posterior_prob[5][:14], label="Size 5")
    plt.plot(np.array(range(14)), posterior_prob[7][:14], label="Size 7")
    plt.plot(np.array(range(14)), posterior_prob[9][:14], label="Size 9")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Days of observation")
    plt.ylabel("$Pr(\mathrm{infected}\/|\/\mathrm{ released})$")
    plt.show()
    code.interact(local=locals())

""" no tracing, no testing """
def simulate_baseline(num_index_cases, initial=True, base_reduction=0.0, frac_vacc=0., vacc_eff=0.,
                      seed=0):
    np.random.seed(seed)
    g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US, initial=initial, frac_vacc=frac_vacc, seed=seed)
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    g0_contacts = draw_contact_generation(g0_cases, base_reduction=base_reduction, frac_vacc=frac_vacc, seed=seed)
    g1_cases = draw_traced_generation_from_contacts(g0_cases, g0_contacts, vacc_eff=vacc_eff, seed=seed)
    n_cases_g1 = np.sum(g1_cases['I_COVID'])
    print("n_cases_g1", n_cases_g1)
    aggregated_infections = aggregate_infections(g1_cases['id_original_case'], num_index_cases, np.sum(g1_cases['I_COVID']))
    downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
    downstream_COVID_sorted = np.flip(np.sort(downstream_COVID))
    top_20 = np.cumsum(downstream_COVID_sorted)[int(len(downstream_COVID) * 0.2)]
    print("SS share %f" % (top_20 / np.sum(downstream_COVID)))
    downstream_asymptoms = aggregated_infections[(g0_cases['I_asymptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_symptoms = aggregated_infections[(g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader_symptoms = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    r0 = np.mean(downstream_COVID)
    nbinom_fit = fit_dist.fit_neg_binom(downstream_COVID)
    k = nbinom_fit[1][1]
    print("frac inf by asymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_asymp'])) / n_cases_g1))
    print("frac inf by presymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_presymp'])) / n_cases_g1))
    print("number of transmission days", np.sum(g0_cases['n_transmission_days']) / n_cases_g0)
    # print(g0_cases)
    # print(g1_cases)

    # plt.hist(downstream_COVID, bins=np.arange(0, downstream_COVID.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ (mean: %f, dispersion: %f)' % (r0, k))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    #
    # plt.hist(downstream_asymptoms, bins=np.arange(0, downstream_asymptoms.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ if asymptoms in $g_0$ (mean: %f)' % np.mean(downstream_asymptoms))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    #
    # plt.hist(downstream_symptoms, bins=np.arange(0, downstream_symptoms.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ if symptoms in $g_0$ (mean: %f)' % np.mean(downstream_symptoms))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    #
    # plt.hist(downstream_superspreader, bins=np.arange(0, downstream_superspreader.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ if superspreader (mean: %f)' % np.mean(downstream_superspreader))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    #
    # plt.hist(downstream_superspreader_symptoms, bins=np.arange(0, downstream_superspreader_symptoms.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ if superspreader and symptoms (mean: %f)' % np.mean(downstream_superspreader_symptoms))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()

    #
    # plt.hist(g0_cases['n_isolation_days'], density=True)
    # plt.xlabel(r'isolation days')
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    # plt.hist([len(list(x)) for x in g0_contacts], density=True)
    # plt.xlabel(r'number of contacts (mean: %f)' % np.mean([len(list(x)) for x in g0_contacts]))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    # plt.hist((g1_cases['n_age']).astype(int), bins=np.arange(0, g1_cases['n_age'].max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'age range $g_1$')
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    # plt.hist((np.array(g0_cases['n_age'])).astype(int), bins=np.arange(0, np.array(g0_cases['n_age']).max() + 1.5) - 0.5,density=True)
    # plt.xlabel(r'age range $g_0$')
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    return ((r0, k), 0., np.sum(downstream_asymptoms) / np.sum(g0_cases['I_asymptoms']), np.sum(downstream_symptoms) / np.sum(g0_cases['I_symptoms']))

age_to_contact_dict = get_contacts_per_age()

if __name__ == "__main__":
    main()
