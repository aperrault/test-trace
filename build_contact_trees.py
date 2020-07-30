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

age_intervals = [(0, 14), (15, 24), (25, 54), (55, 64), (65, np.inf)]
age_vector_US = np.array([0.1862, 0.1312, 0.3929, 0.1294, 0.1603])
age_vector_US = age_vector_US / np.sum(age_vector_US)
false_positive_rate = 0.005
frac_asymptoms = 0.16
asymp_infectiousness = 0.1
""" relative transmission rate of asymptoms individual """
days_infectious_before_symptoms = 2
days_infectious_after_symptoms = 5
total_infectious_days = days_infectious_before_symptoms + days_infectious_after_symptoms
asymp_quarantine_length = 14
symp_quarantine_length = 10
symptoms_household_SAR = 0.2
symptoms_external_SAR = 0.06
superspreader_SAR_ratio = 10
frac_infs_from_SS = 0.9
index_R0 = 2.548934288671604
raw_R0 = 2.5
days_to_track = 10
quarantine_dropout_rate = 0.05
quarantine_dropout_rate_pos_test = 0.0
quarantine_dropout_rate_not_released = 0.025
quarantine_dropout_rate_neg_test = 0.10
length_of_sympts_to_track = 30
symptom_false_positive = 0.01

frac_SS_cal = 0.094
phys_mult_cal = 1 + np.log(1 + np.exp(-0.28))
SS_mult_cal = 1 + np.log(1 + np.exp(30.))
base_household_cal = 0.045
base_external_cal = 0.012

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
def true_positive_test_rate(test_time_rel_to_symptoms):
    if test_time_rel_to_symptoms <= -3:
        return 1.0
    if test_time_rel_to_symptoms == -2:
        return 0.95
    if test_time_rel_to_symptoms == -1:
        return 0.7
    if test_time_rel_to_symptoms == 0:
        return 0.4
    else:
        return 0.25


@jit(nopython=True, parallel=True)
def calc_test_results(num_individuals, test_time_rel_to_symptoms, I_COVID):
    test_results = np.zeros(num_individuals)
    for i in prange(num_individuals):
        test_results[i] = calc_test_single(test_time_rel_to_symptoms[i], I_COVID[i])
    return test_results

@njit
def calc_test_single(test_time_rel_to_symptoms, I_COVID):
    if I_COVID:
        return np.random.binomial(n=1, p=1 - true_positive_test_rate(int(round(test_time_rel_to_symptoms))))
    else:
        return 0

""" draw properties of seed index cases """
def draw_seed_index_cases(num_individuals, age_vector, cases_contacted=1.0,
                          t_exposure=None, fpr=false_positive_rate, initial=False, skip_days=False,
                          random_testing=False, test_freq=0., test_delay=0.,
                          frac_SS=frac_SS_cal):
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
    params['t_incubation'] = np.maximum(scipy.stats.lognorm.rvs(s=0.29878047, loc=-1.37231096, scale=6.57231006, size=num_individuals), 0) + t_exposure
    params['t_incubation_infect'] = np.maximum(params['t_incubation'] - days_infectious_before_symptoms, 0)
    if initial:
        params['I_self_isolate'] = np.ones(num_individuals) * params['I_symptoms']
    else:
        params['I_self_isolate'] = scipy.stats.bernoulli.rvs(p=0.9, size=num_individuals) * params['I_symptoms']
    params['t_false_positive'] = np.random.geometric(p=symptom_false_positive, size=num_individuals)
    params['t_self_isolate'] = scipy.stats.uniform.rvs(loc=days_infectious_before_symptoms, scale=3, size=num_individuals) + params['t_incubation_infect']
    if skip_days:
        params['n_transmission_days'] = (
            params['I_asymptoms'] * total_infectious_days +
            params['I_symptoms'] * (1 - params['I_self_isolate']) * total_infectious_days +
            params['I_symptoms'] * params['I_self_isolate'] * (params['t_self_isolate'] - params['t_incubation_infect']))
        params['n_quarantine_days'] = np.zeros(num_individuals)
        params['n_isolation_days'] = params['I_self_isolate'] * symp_quarantine_length
        params['n_tests'] = np.zeros(num_individuals)
    else:
        (params['n_quarantine_days'], params['n_transmission_days'], params['n_isolation_days'], params['n_tests'], params['n_monitoring_days']) = get_transmission_days(
            t_exposure=params['t_exposure'], t_last_exposure=params['t_last_exposure'],
            I_isolation=params['I_self_isolate'], t_isolation=params['t_self_isolate'],
            t_infectious=params['t_incubation_infect'],
            I_symptoms=params['I_symptoms'], t_symptoms=params['t_incubation'],
            I_random_testing=random_testing, test_freq=test_freq,
            test_delay=test_delay)

    params['id_original_case'] = (np.ones(num_individuals) * -1).astype(int)
    params['id_infected_by'] = (np.ones(num_individuals) * -1).astype(int)
    params['I_superspreader'] = scipy.stats.bernoulli.rvs(p=frac_SS, size=num_individuals) * params['I_COVID']
    if initial:
        params['I_test_positive'] = np.ones(num_individuals)
    else:
        params['I_test_positive'] = calc_test_results(num_individuals,
                                                     params['t_self_isolate'] - params['t_incubation_infect'],
                                                     params['I_COVID'])
    params['I_infected_by_non_sympt'] = np.ones(num_individuals) * -1
    params['n_quarantine_days_of_uninfected'] = 0.
    return params

@njit
def draw_contacts(contact_days, n_transmission_days, t_incubation_infect, index, base_reduction=0.0, get_dummy=False):
    contact_list = List()
    contacts = List()
    selection = np.random.choice(len(contact_days))
    for day in range(n_transmission_days):
        contacts.append(contact_days[selection])
    if n_transmission_days > 0 and not get_dummy:
        mask = np.random.binomial(n=1, p=(1 - base_reduction), size=len(contacts[0]))
    for day in range(n_transmission_days):
        for (i, contact) in enumerate(contacts[day]):
            # if (day == 0 or not contact[3] or np.random.binomial(n=1, p=(1. - 5./7.) ** day)) or get_dummy:
            # if (day == 0 or not contact[3] or np.random.binomial(n=1, p=0.5)) or get_dummy:
            if day == 0 or not contact[3] or get_dummy:
                if not get_dummy:
                    if not (mask[i] or contact[2]):
                        continue
                """ contact format: (age of contact, gender, is_household, is_daily,
                                     t_exposure, index of infector, t_last_exposure, is_physical) """
                contact_ext = np.zeros(8)
                contact_ext[0] = contact[0]
                contact_ext[1] = contact[1]
                contact_ext[2] = contact[2]
                contact_ext[3] = contact[3]
                if contact[3]:
                    day_of_infection = random.randrange(0, n_transmission_days)
                contact_ext[4] = day_of_infection + t_incubation_infect
                contact_ext[5] = index
                if contact[3]:
                    contact_ext[6] = n_transmission_days - 1
                else:
                    contact_ext[6] = day
                contact_ext[7] = contact[4]
                contact_list.append(contact_ext)
    return contact_list

@njit
def draw_all_contacts(contacts_by_age, n_transmission_days, t_incubation_infect, num_individuals, base_reduction=0.0):
    contact_dict = List()
    for i in range(num_individuals):
        a = draw_contacts(contact_days=contacts_by_age[0],
                          n_transmission_days=n_transmission_days[0],
                          t_incubation_infect=t_incubation_infect[0],
                          index=i, get_dummy=True)
        if len(a) > 0:
            break
    for i in range(num_individuals):
        contact_dict.append(a)
    for i in range(num_individuals):
        contact_dict[i] = draw_contacts(contact_days=contacts_by_age[i],
                                        n_transmission_days=n_transmission_days[i],
                                        t_incubation_infect=t_incubation_infect[i],
                                        index=i, base_reduction=base_reduction)
    return contact_dict

""" draw index cases assuming uniform attack rate. draw contacts from POLYMOD """
def draw_contact_generation(index_cases, base_reduction=0.0):
    num_individuals = len(index_cases['I_COVID'])
    if num_individuals == 0:
        return []
    contacts_by_age = [age_to_contact_dict[index_cases['n_age'][i]] for i in range(num_individuals)]
    contact_dict = draw_all_contacts(contacts_by_age=contacts_by_age,
                                     n_transmission_days=index_cases['n_transmission_days'],
                                     t_incubation_infect=index_cases['t_incubation_infect'],
                                     num_individuals=num_individuals,
                                     base_reduction=base_reduction)
    return contact_dict

@jit(nopython=True, parallel=True)
def fill_QII(t_exposure, t_last_exposure, trace_delay, test_delay, t_self_isolate,
             I_self_isolate, t_incubation_infect,
             t_incubation, I_symptoms,
             id_infected_by, num_g1_cases, g0_I_self_isolate, g0_t_self_isolate,
             g0_I_test_positive, g0_I_symptoms, g0_I_contacted,
             g0_I_superspreader, trace=False, trace_superspreader=False,
             monitor_non_SS=False, bidirectional_SS=False, tfn=0.0, ttq=False,
             quarantine_by_parent_case_release=False,
             quarantine_early_release_day=0, early_release=False,
             wait_before_testing=0, ttq_double=False,
             monitor=False):
    n_quarantine_days = np.zeros(num_g1_cases)
    n_transmission_days = np.zeros(num_g1_cases)
    n_isolation_days = np.zeros(num_g1_cases)
    n_monitoring_days = np.zeros(num_g1_cases)
    n_tests = np.zeros(num_g1_cases)

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
            and (trace or (g0_I_superspreader[parent_case] and trace_superspreader)) and (tfn < 0.00001 or np.random.binomial(n=1, p=1 - tfn) > 0.99)):
            secondary_cases_traced += 1
            t_quarantine_start = g0_t_self_isolate[parent_case] + trace_delay + test_delay
            if early_release and quarantine_by_parent_case_release[parent_case]:
                t_quarantine_end = t_quarantine_start + quarantine_early_release_day
                if monitor:
                    t_monitor = t_quarantine_start + quarantine_early_release_day
                    t_monitor_end = max(t_last_exposure[i] + asymp_quarantine_length, t_quarantine_start)
            else:
                t_quarantine_end = max(t_last_exposure[i] + asymp_quarantine_length, t_quarantine_start)
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=t_quarantine_start,
                                     t_quarantine_end=t_quarantine_end,
                                     I_quarantine=True,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     I_monitor=monitor,
                                     t_monitor=t_quarantine_start,
                                     t_monitor_end=t_quarantine_end,
                                     I_test=ttq,
                                     t_test_day=t_quarantine_start + wait_before_testing,
                                     test_delay=test_delay,
                                     I_double_test=ttq_double,
                                     wait_before_testing=wait_before_testing,
                                     I_early_release=early_release,
                                     early_release_day=quarantine_early_release_day)
            n_quarantine_days[i] += tmp[0]
            n_isolation_days[i] += tmp[1]
            n_transmission_days[i] += tmp[2]
            n_tests[i] += tmp[3]
            n_monitoring_days[i] += tmp[4]
        elif (g0_I_test_positive[parent_case] and g0_I_self_isolate[parent_case]
              and g0_I_contacted[parent_case] and monitor_non_SS
              and not g0_I_superspreader[parent_case]):
            monitoring_delay = trace_delay
            if I_bidirectionally_traced[parent_case]:
                monitoring_delay = earliest_self_isolates[parent_case] + 2 * (trace_delay + test_delay)
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=0,
                                     t_quarantine_end=0,
                                     I_quarantine=False,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     I_monitor=True,
                                     t_monitor=monitoring_delay,
                                     test_delay=test_delay)
            n_quarantine_days[i] += tmp[0]
            n_isolation_days[i] += tmp[1]
            n_transmission_days[i] += tmp[2]
            n_tests[i] += tmp[3]
            n_monitoring_days[i] += tmp[4]
            secondary_cases_monitored += 1
        elif (bidirectional_SS and g0_I_test_positive[parent_case] and
              g0_I_superspreader[parent_case] and
              g0_I_symptoms[parent_case] and I_bidirectionally_traced[parent_case] == 1):
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=earliest_self_isolates[parent_case] + 2 * (trace_delay + test_delay),
                                     t_quarantine_end=-1,  ####!!!!
                                     I_quarantine=True,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     test_delay=test_delay)
            n_quarantine_days[i] += tmp[0]
            n_isolation_days[i] += tmp[1]
            n_transmission_days[i] += tmp[2]
            n_tests[i] += tmp[3]
            n_monitoring_days[i] += tmp[4]
        else:
            tmp = calculate_QII_days(t_exposure=t_exposure[i],
                                     t_last_exposure=t_last_exposure[i],
                                     t_quarantine=0,
                                     I_quarantine=False,
                                     t_quarantine_end=0,
                                     t_isolation=t_self_isolate[i],
                                     I_isolation=I_self_isolate[i],
                                     t_infectious=t_incubation_infect[i],
                                     t_symptoms=t_incubation[i],
                                     I_symptoms=I_symptoms[i],
                                     I_monitor=False,
                                     t_monitor=-1,
                                     I_test=False,
                                     t_test_day=-1,
                                     test_delay=test_delay)
            n_quarantine_days[i] += tmp[0]
            n_isolation_days[i] += tmp[1]
            n_transmission_days[i] += tmp[2]
            n_tests[i] += tmp[3]
            n_monitoring_days[i] += tmp[4]
    return (n_quarantine_days, n_transmission_days, n_isolation_days, secondary_cases_traced, secondary_cases_monitored, n_monitoring_days,
            n_tests)

# positives = [0]
# negatives = [0]
# days_rel_to_sympts = []

@njit
def calculate_QII_days(t_exposure, t_last_exposure,
                       t_infectious,
                       I_quarantine=False, t_quarantine=0., t_quarantine_end=0.,
                       I_isolation=False, t_isolation=0.,
                       I_symptoms=False, t_symptoms=0.,
                       I_monitor=False, t_monitor=0., t_monitor_end=0.,
                       I_test=False, t_test_day=0., test_delay=0.,
                       I_random_testing=False, test_freq=0.,
                       I_double_test=False, wait_before_testing=0,
                       early_release_day=0, I_early_release=False):
    # if I_random_testing and test_freq > 0.0:
    #     test_interval = 1 / test_freq
    # else:
    #     test_interval = 10000
    # first_test = t_infectious + np.random.uniform(0.0, test_interval)

    quarantine_days = 0
    isolation_days = 0
    transmission_days = 0
    monitoring_days = 0
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
    while not (day >= t_quarantine_end and day >= t_isolation_end and day >= t_isolation_end and day >= t_infectious_end):
        # code.interact(local=locals())
        # symptoms are developed during quarantine or already exist when quarantine started
        if I_symptoms and day >= t_symptoms and ((I_quarantine and t_quarantine <= day < t_quarantine_end) or (I_monitor and t_monitor <= day < t_monitor_end)) and not symptoms_observed:
            if I_isolation:
                t_isolation = min(t_isolation, day)
            else:
                t_isolation = day
                I_isolation = True
            t_isolation_end = t_isolation + symp_quarantine_length
            t_quarantine_end = min(t_quarantine_end, t_isolation_end)
            symptoms_observed = True

        # is this person quarantined today? do they drop out?
        if t_infectious <= day < t_infectious_end:
            if (not I_quarantine or not (t_quarantine <= day < t_quarantine_end)) and (not I_isolation or not (t_isolation <= day < t_isolation_end)):
                transmission_days += 1
            elif (I_isolation and t_isolation <= day < t_isolation_end) or (I_quarantine and t_quarantine <= day < t_quarantine_end):
                isolation_days += 1
        elif I_quarantine and t_quarantine <= day < t_quarantine_end:
            if I_isolation and t_isolation <= day < t_isolation_end:
                isolation_days += 1
            else:
                quarantine_dropout_chance = quarantine_dropout_rate
                if I_early_release and day >= early_release_day:
                    quarantine_dropout_chance = quarantine_dropout_rate_not_released
                if test_results_day_exists and test_results_day <= day < test_results_day + 1:
                    if I_test_positive:
                        quarantine_dropout_chance = quarantine_dropout_rate_pos_test
                    else:
                        quarantine_dropout_chance = quarantine_dropout_rate_neg_test
                if np.random.binomial(n=1, p=quarantine_dropout_chance):
                    t_quarantine_end = day
                    t_monitor_end = day
                else:
                    quarantine_days += 1
        elif I_isolation and t_isolation <= day < t_isolation_end:
            isolation_days += 1
        elif I_monitor and t_monitor <= day < t_monitor_end:
            monitoring_days += 1

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
    return (quarantine_days, isolation_days, transmission_days, n_tests, monitoring_days)

@njit
def draw_infections_from_contacts(num_individuals, g0_COVID, g0_symptoms,
                                  g0_original_case, g0_superspreader,
                                  g0_test_positive, g0_incubation_infect,
                                  g0_t_self_isolate, g0_I_contacted,
                                  g0_contacts, trace_delay=0, test_delay=0,
                                  trace=False, trace_superspreader=False,
                                  ttq=False, early_release=False,
                                  tfn=0.0, wait_before_testing=0, ttq_double=False,
                                  household_SAR=base_household_cal, external_SAR=base_external_cal,
                                  SS_mult=SS_mult_cal, phys_mult=phys_mult_cal,
                                  monitor=False):
    n_age = List()
    t_exposure = List()
    t_last_exposure = List()
    original_case = List()
    infected_by = List()
    I_COVID = List()
    infected_by_non_sympt = List()
    quarantine_days_of_uninfected = 0
    tests_of_uninfected = 0
    monitoring_days_of_uninfected = 0
    # SAR_SS_household = max(min(frac_infs_from_SS * symptoms_household_SAR / ((1 + frac_infs_from_SS) / (1 + 1 / freq_superspreaders)), 1), 0)
    # SAR_SS_ext = max(min(frac_infs_from_SS * symptoms_external_SAR / ((1 + frac_infs_from_SS) / (1 + 1 / freq_superspreaders)), 1), 0)
    # SAR_reg_household =  max(min((symptoms_household_SAR - SAR_SS_household * freq_superspreaders) / (1 - freq_superspreaders), 1), 0)
    # SAR_reg_ext =  max(min((symptoms_external_SAR - SAR_SS_ext * freq_superspreaders) / (1 - freq_superspreaders), 1), 0)

    uninfected_source = List()
    uninfected_exposure = List()

    household_exposures = 0
    household_infections = 0
    external_exposures = 0
    external_infections = 0

    num_downstream_contacts_by_id = np.zeros(num_individuals)

    for i in range(num_individuals):
        if not g0_COVID[i]:
            continue
        for j in range(len(g0_contacts[i])):
            num_downstream_contacts_by_id[i] += 1
            infected = False
            SAR = 1.0
            if g0_contacts[i][j][2]:
                SAR = household_SAR
                household_exposures += 1
            else:
                SAR = external_SAR
                external_exposures += 1
            if g0_contacts[i][j][8]:
                SAR = SAR * phys_mult
            if g0_superspreader[i]:
                SAR = SAR * SS_mult
            if not g0_symptoms[i]: # or g0_contacts[i][j][6] < days_infectious_before_symptoms:
                SAR = SAR * asymp_infectiousness
            SAR = max(min(SAR, 1.), 0.)
            infected = np.random.binomial(n=1, p=SAR)
            if infected:
                if g0_contacts[i][j][2]:
                    household_infections += 1
                else:
                    external_infections += 1
                if g0_contacts[i][j][4] < g0_incubation_infect[i] + days_infectious_before_symptoms or not g0_symptoms[i]:
                    infected_by_non_sympt.append(True)
                else:
                    infected_by_non_sympt.append(False)
                n_age.append(g0_contacts[i][j][0])
                t_exposure.append(g0_contacts[i][j][4])
                t_last_exposure.append(g0_contacts[i][j][6])
                infected_by.append(g0_contacts[i][j][5])
                I_COVID.append(1)
                if g0_original_case[i] == -1:
                    original_case.append(g0_contacts[i][j][5])
                else:
                    original_case.append(g0_original_case[i])
            else:
                if (g0_I_contacted[i] and trace) and (tfn < 0.00001 or np.random.binomial(n=1, p=1 - tfn) > 0.99):
                    # trace_start = g0_t_self_isolate[i] + trace_delay + test_delay
                    # quarantine_end = asymp_quarantine_length + g0_contacts[i][j][6]
                    uninfected_source.append(i)
                    uninfected_exposure.append(g0_contacts[i][j][6])
    return (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID,
            quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
            infected_by_non_sympt, uninfected_source, uninfected_exposure,
            num_downstream_contacts_by_id, household_infections / household_exposures,
            external_infections / external_exposures)

@jit(nopython=True, parallel=True)
def calc_symptoms_by_day(g1_I_symptoms, g1_infected_by, g1_t_incubation, g1_false_positive,
                         g0_t_self_isolate, trace_delay, test_delay,
                         uninf_g1_source, uninf_g1_false_positive):
    symptoms_by_day = np.zeros((len(g0_t_self_isolate), days_to_track))

    for i in range(len(g1_I_symptoms)):
        parent_case = int(g1_infected_by[i])
        time_of_symptoms_rel_to_monitoring = g1_false_positive[i]
        if g1_I_symptoms[i]:
            time_of_symptoms_rel_to_monitoring = min(g1_false_positive[i], g1_t_incubation[i] - (g0_t_self_isolate[parent_case] + trace_delay + test_delay))
        for j in range(days_to_track):
            if j < time_of_symptoms_rel_to_monitoring:
                symptoms_by_day[parent_case, j] += 1

    for i in range(len(uninf_g1_source)):
        for j in range(days_to_track):
            if j < uninf_g1_false_positive[i]:
                symptoms_by_day[parent_case, j] += 1

    return symptoms_by_day

@njit
def calc_quarantine_uninfected(trace_start, quarantine_end,
                               wait_before_testing=0, test_delay=0,
                               early_release_day=0, early_release=False,
                               n_consec_test=0, test_release=False,
                               I_monitor=False, monitor_start=0,
                               monitor_end=0, false_positive_day=0):
    day = trace_start
    tests = 0
    consec_negative_tests = 0
    quarantine_dropout_chance = quarantine_dropout_rate
    quarantine_days = 0
    monitor_days = 0
    end = quarantine_end
    test_results_day = 0
    if I_monitor:
        end = max(quarantine_end, monitor_end)
    while day < end:
        if trace_start <= day < quarantine_end:
            quarantine_days += 1
        elif I_monitor and monitor_start <= day < monitor_end:
            monitor_days += 1
        if test_release and day >= test_results_day:
            if consec_negative_tests >= n_consec_test:
                quarantine_end = day
            quarantine_dropout_chance = quarantine_dropout_rate_neg_test
        if trace_start <= day < quarantine_end and np.random.binomial(n=1, p=quarantine_dropout_chance):
            quarantine_end = day
            if I_monitor:
                monitor_end = day
        if test_release and day - test_results_day >= wait_before_testing and trace_start <= day < quarantine_end:
            test_results_day = day + test_delay
            tests += 1
            consec_negative_tests += 1
        if early_release and day >= early_release_day:
            quarantine_dropout_chance = quarantine_dropout_rate_not_released
        day += 1
    # print(n_consec_test, test_release, initial_quarantine_length, quarantine_days)
    return (quarantine_days, tests, monitor_days)

@jit(nopython=True, parallel=True)
def calc_quarantine_days_of_uninfected(uninf_g1_source, uninf_g1_exposure_day,
                                       g1_I_symptoms, g1_t_symptoms,
                                       g0_t_self_isolate, g0_test_positive,
                                       g0_I_contacted, g0_superspreader, ttq=False,
                                       ttq_double=False, trace=False,
                                       trace_delay=0, test_delay=0, wait_before_testing=0,
                                       early_release=False,
                                       quarantine_by_parent_case_release=np.zeros((1)),
                                       quarantine_early_release_day=False,
                                       monitor=False):
    quarantine_days_of_uninfected = 0.
    tests_of_uninfected = 0.
    monitoring_days_of_uninfected = 0.
    total_contacts = 0.
    released_contacts = 0.
    for i in prange(len(uninf_g1_source)):
        parent_case = int(uninf_g1_source[i])
        if g0_I_contacted[parent_case] and trace:
            total_contacts += 1
            trace_start = g0_t_self_isolate[parent_case] + trace_delay + test_delay
            quarantine_end = asymp_quarantine_length + uninf_g1_exposure_day[i]
            monitor_start = trace_start
            monitor_end = quarantine_end
            if early_release:
                if not quarantine_by_parent_case_release[parent_case]:
                    tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                     quarantine_end=quarantine_end,
                                                     early_release=True,
                                                     early_release_day=quarantine_early_release_day,
                                                     I_monitor=monitor,
                                                     monitor_start=monitor_start,
                                                     monitor_end=monitor_end)
                    quarantine_days_of_uninfected += tmp[0]
                    tests_of_uninfected += tmp[1]
                    monitoring_days_of_uninfected += tmp[2]
                else:
                    tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                     quarantine_end=min(asymp_quarantine_length + uninf_g1_exposure_day[i],
                                                                        trace_start + quarantine_early_release_day),
                                                     I_monitor=monitor,
                                                     monitor_start=monitor_start,
                                                     monitor_end=monitor_end)
                    quarantine_days_of_uninfected += tmp[0]
                    tests_of_uninfected += tmp[1]
                    monitoring_days_of_uninfected += tmp[2]
                    released_contacts += 1
            elif ttq:
                tmp = calc_quarantine_uninfected(trace_start=trace_start, quarantine_end=quarantine_end,
                                                 wait_before_testing=wait_before_testing, test_delay=test_delay,
                                                 n_consec_test=1, test_release=True, I_monitor=monitor,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end
                                                 )
                quarantine_days_of_uninfected += tmp[0]
                tests_of_uninfected += tmp[1]
                monitoring_days_of_uninfected += tmp[2]
            elif ttq_double:
                tmp = calc_quarantine_uninfected(trace_start=trace_start, quarantine_end=quarantine_end,
                                                 wait_before_testing=wait_before_testing, test_delay=test_delay,
                                                 n_consec_test=2, test_release=True, I_monitor=monitor,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end)
                quarantine_days_of_uninfected += tmp[0]
                tests_of_uninfected += tmp[1]
                monitoring_days_of_uninfected += tmp[2]
            else:
                tmp = calc_quarantine_uninfected(trace_start=trace_start,
                                                 quarantine_end=quarantine_end, I_monitor=monitor,
                                                 monitor_start=monitor_start,
                                                 monitor_end=monitor_end)
                quarantine_days_of_uninfected += tmp[0]
                tests_of_uninfected += tmp[1]
                monitoring_days_of_uninfected += tmp[2]
    if early_release:
        release_pct = released_contacts / total_contacts
        print("percent of contacts released")
        print(release_pct)
    return (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected)

def draw_traced_generation_from_contacts(g0_cases, contacts, trace=False,
                                         trace_superspreader=False, monitor_non_SS=False,
                                         bidirectional_SS=False,
                                         trace_delay=0, test_delay=0, trace_false_negative=0.0,
                                         ttq=False, early_release=False,
                                         early_release_day=0, early_release_threshold=0.0,
                                         wait_before_testing=0,
                                         ttq_double=False,
                                         household_SAR=base_household_cal, external_SAR=base_external_cal,
                                         SS_mult=SS_mult_cal, phys_mult=phys_mult_cal,
                                         monitor=False):
    (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID,
     quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
     infected_by_non_sympt, uninfected_source, uninfected_exposure,
     num_downstream_contacts_by_id, household_SAR, external_SAR) = draw_infections_from_contacts(
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
         trace=trace,
         trace_superspreader=trace_superspreader,
         ttq=ttq,
         ttq_double=ttq_double,
         early_release=early_release,
         tfn=trace_false_negative,
         wait_before_testing=wait_before_testing,
         household_SAR=household_SAR, external_SAR=external_SAR,
         SS_mult=SS_mult, phys_mult=phys_mult,
         monitor=monitor)

    t_exposure = np.array(t_exposure)
    num_g1_cases = len(n_age)
    g1_cases = draw_seed_index_cases(num_individuals=num_g1_cases,
                                     age_vector=age_vector_US,
                                     t_exposure=t_exposure,
                                     skip_days=True)
    g1_cases['t_last_exposure'] = np.array(t_last_exposure)
    g1_cases['n_age'] = np.array(n_age)
    g1_cases['id_original_case'] = np.array(original_case)
    g1_cases['id_infected_by'] = np.array(infected_by)
    g1_cases['I_COVID'] = np.array(I_COVID)
    g1_cases['I_infected_by_non_sympt'] = np.array(infected_by_non_sympt)
    g1_cases['household_SAR'] = household_SAR
    g1_cases['external_SAR'] = external_SAR

    quarantine_by_parent_case_release = np.zeros(len(g0_cases['I_COVID']))
    quarantine_early_release_day = early_release_day + 1

    uninf_false_positive = np.random.geometric(p=symptom_false_positive, size=len(uninfected_source))
    if early_release_day is None:
        early_release_day = 0
    if early_release:
        symptoms_by_day = calc_symptoms_by_day(
            g1_I_symptoms=g1_cases['I_symptoms'],
            g1_infected_by=g1_cases['id_infected_by'],
            g1_t_incubation=g1_cases['t_incubation'],
            g1_false_positive=g1_cases['t_false_positive'],
            g0_t_self_isolate=g0_cases['t_self_isolate'],
            trace_delay=trace_delay, test_delay=test_delay,
            uninf_g1_source=uninfected_source,
            uninf_g1_false_positive=uninf_false_positive)

        percentage_by_day = np.nan_to_num(symptoms_by_day / (num_downstream_contacts_by_id.reshape(-1, 1)))
        quarantine_by_parent_case_release = (percentage_by_day[:, early_release_day] < early_release_threshold)
        (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected) = calc_quarantine_days_of_uninfected(
            uninf_g1_source=uninfected_source,
            uninf_g1_exposure_day=uninfected_exposure,
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
            quarantine_by_parent_case_release=quarantine_by_parent_case_release,
            quarantine_early_release_day=quarantine_early_release_day,
            monitor=monitor,
            symptom_false_positive=uninf_false_positive)
    else:
        (quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected) = calc_quarantine_days_of_uninfected(
            uninf_g1_source=uninfected_source,
            uninf_g1_exposure_day=uninfected_exposure,
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
            trace_delay=trace_delay,
            test_delay=test_delay,
            quarantine_by_parent_case_release=quarantine_by_parent_case_release,
            quarantine_early_release_day=quarantine_early_release_day,
            monitor=monitor,
            symptom_false_positive=uninf_false_positive)

    (g1_cases['n_quarantine_days'], g1_cases['n_transmission_days'], g1_cases['n_isolation_days'],
     secondary_cases_traced, secondary_cases_monitored, g1_cases['n_monitoring_days'],
     g1_cases['n_tests']) = fill_QII(
        t_exposure=t_exposure, t_last_exposure=g1_cases['t_last_exposure'],
        trace_delay=trace_delay, test_delay=test_delay,
        t_self_isolate=g1_cases['t_self_isolate'],
        I_self_isolate=g1_cases['I_self_isolate'],
        t_incubation_infect=g1_cases['t_incubation_infect'],
        t_incubation=g1_cases['t_incubation'],
        I_symptoms=g1_cases['I_symptoms'],
        id_infected_by=g1_cases['id_infected_by'],
        num_g1_cases=num_g1_cases,
        g0_I_contacted=g0_cases['I_contacted'],
        g0_I_self_isolate=g0_cases['I_self_isolate'],
        g0_t_self_isolate=g0_cases['t_self_isolate'],
        g0_I_test_positive=g0_cases['I_test_positive'],
        g0_I_symptoms=g0_cases['I_symptoms'],
        g0_I_superspreader=g0_cases['I_superspreader'],
        trace=trace, trace_superspreader=trace_superspreader,
        monitor_non_SS=monitor_non_SS,
        bidirectional_SS=bidirectional_SS,
        tfn=trace_false_negative,
        ttq=ttq,
        ttq_double=ttq_double,
        early_release=early_release,
        quarantine_by_parent_case_release=quarantine_by_parent_case_release,
        quarantine_early_release_day=early_release_day,
        wait_before_testing=wait_before_testing,
        monitor=monitor)
    print("%i secondary cases traced" % secondary_cases_traced)
    print("%i secondary cases monitored" % secondary_cases_monitored)
    g1_cases['n_quarantine_days_of_uninfected'] = quarantine_days_of_uninfected
    g1_cases['n_tests_of_uninfected'] = tests_of_uninfected
    g1_cases['n_monitoring_days_of_uninfected'] = monitoring_days_of_uninfected
    g1_cases['secondary_cases_traced'] = secondary_cases_traced
    g1_cases['secondary_cases_monitored'] = secondary_cases_monitored
    return g1_cases

@jit(nopython=True, parallel=True)
def get_transmission_days(t_exposure, t_last_exposure,
                          I_isolation, t_isolation,
                          t_infectious,
                          I_symptoms, t_symptoms,
                          test_freq=0,
                          test_delay=0.,
                          I_random_testing=False):
    n_quarantine_days = np.zeros(len(t_exposure))
    n_transmission_days = np.zeros(len(t_exposure))
    n_isolation_days = np.zeros(len(t_exposure))
    n_tests = np.zeros(len(t_exposure))
    n_monitoring_days = np.zeros(len(t_exposure))

    for i in prange(len(t_exposure)):
        (n_quarantine_days[i], n_isolation_days[i], n_transmission_days[i], n_tests[i], n_monitoring_days[i]) = calculate_QII_days(t_exposure=t_exposure[i], t_last_exposure=t_last_exposure[i],
                                                                                                 I_isolation=I_isolation[i], t_isolation=t_isolation[i],
                                                                                                 t_infectious=t_infectious[i],
                                                                                                 I_symptoms=I_symptoms[i], t_symptoms=t_symptoms[i],
                                                                                                 test_freq=test_freq, test_delay=test_delay,
                                                                                                 I_random_testing=I_random_testing)
    return (n_quarantine_days, n_transmission_days, n_isolation_days, n_tests, n_monitoring_days)

def simulate_testing(num_index_cases, test_freq, test_delay):
    g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US,
                                     random_testing=True, test_freq=test_freq, test_delay=test_delay)
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    g0_contacts = draw_contact_generation(g0_cases)
    g1_cases = draw_traced_generation_from_contacts(g0_cases, g0_contacts)
    n_cases_g1 = np.sum(g1_cases['I_COVID'])
    aggregated_infections = np.zeros(num_index_cases)
    for i in range(int(n_cases_g1)):
        aggregated_infections[int(g1_cases['id_original_case'][i])] += 1
    downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
    downstream_asymptoms = aggregated_infections[(g0_cases['I_asymptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_symptoms = aggregated_infections[(g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader_symptoms = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    r0 = np.mean(downstream_COVID)
    nbinom_fit = fit_dist.fit_neg_binom(downstream_COVID)
    k = nbinom_fit[1][1]
    print("frac inf by non-symptoms: %f" % (float(np.sum(g1_cases['I_infected_by_non_sympt'])) / n_cases_g1))
    return ((r0, k), 0., np.sum(downstream_asymptoms) / np.sum(g0_cases['I_asymptoms']), np.sum(downstream_symptoms) / np.sum(g0_cases['I_symptoms']), 1 - (n_cases_g1 / n_cases_g0) / raw_R0)

@jit(nopython=True,parallel=True)
def aggregate_infections(case_ids, n_cases_g0, n_cases_g1):
    aggregated_infections = np.zeros(n_cases_g0)
    for i in prange(n_cases_g1):
        aggregated_infections[int(case_ids[i])] += 1
    return aggregated_infections

# @njit
# def calc_stats(percentage_by_day, cutoffs, eventual_infections, num_downstream_contacts_by_id):
#     cutoffs = [0., 0.01, 0.02, 0.03]
#     num_cutoffs = len(cutoffs) + 1
#     samples = List()
#     sample_mean = np.zeros((num_cutoffs, np.shape(percentage_by_day)[1]))
#     sample_std = np.zeros((num_cutoffs, np.shape(percentage_by_day)[1]))
#     for i in range(num_cutoffs):
#         samples.append(List())
#     for i in range(np.shape(percentage_by_day)[1]):
#         for j in range(num_cutoffs):
#             samples[j].append(List())
#     for j in range(num_cutoffs):
#         for cluster in range(np.shape(percentage_by_day)[0]):
#             if num_downstream_contacts_by_id[cluster] < 1:
#                 continue
#             for day in range(np.shape(percentage_by_day)[1]):
#                 if j < len(cutoffs) and percentage_by_day[cluster][day] > cutoffs[j]:
#                     samples[j][day].append(eventual_infections[cluster])
#                 elif percentage_by_day[cluster][day] < 1e-6:
#                     samples[j][day].append(eventual_infections[cluster])
#
#     for j in range(num_cutoffs):
#         for day in range(np.shape(percentage_by_day)[1]):
#             sample_mean[j][day] = np.mean(samples[j][day])
#             sample_std[j][day] = np.std(samples[j][day])
#     return(sample_mean, sample_std)

def draw_samples(num_index_cases, trace_delay, test_delay, wait_before_testing,
                 trace_false_negative, cases_contacted):
    g0_cases = draw_seed_index_cases(num_individuals=num_index_cases, cases_contacted=cases_contacted,
                                     age_vector=age_vector_US, initial=True)
    g0_contacts = draw_contact_generation(g0_cases)
    (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID,
     quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
     infected_by_non_sympt, uninfected_source, uninfected_exposure,
     num_downstream_contacts_by_id, household_SAR, external_SAR) = draw_infections_from_contacts(
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
        tfn=trace_false_negative,
        wait_before_testing=wait_before_testing)

    t_exposure = np.array(t_exposure)
    num_g1_cases = len(n_age)
    g1_cases = draw_seed_index_cases(num_individuals=num_g1_cases,
                                     age_vector=age_vector_US,
                                     t_exposure=t_exposure,
                                     skip_days=True)
    g1_cases['t_last_exposure'] = np.array(t_last_exposure)
    g1_cases['n_age'] = np.array(n_age)
    g1_cases['id_original_case'] = np.array(original_case)
    g1_cases['id_infected_by'] = np.array(infected_by)
    g1_cases['I_COVID'] = np.array(I_COVID)
    g1_cases['I_infected_by_non_sympt'] = np.array(infected_by_non_sympt)
    g1_cases['household_SAR'] = household_SAR
    g1_cases['external_SAR'] = external_SAR

    quarantine_by_parent_case_release = np.zeros(len(g0_cases['I_COVID']))

    symptoms_by_day = calc_symptoms_by_day(
        g1_I_symptoms=g1_cases['I_symptoms'],
        g1_t_symptoms=g1_cases['t_incubation'],
        g1_infected_by=g1_cases['id_infected_by'],
        g0_t_self_isolate=g0_cases['t_self_isolate'],
        trace_delay=trace_delay, test_delay=test_delay,
        uninf_g1_source=uninfected_source)
    aggregated_infections = aggregate_infections(g1_cases['id_original_case'], num_index_cases, np.sum(g1_cases['I_COVID']))
    eventual_infections = aggregated_infections / num_downstream_contacts_by_id
    percentage_by_day = np.nan_to_num(symptoms_by_day / (num_downstream_contacts_by_id.reshape(-1, 1)))
    cutoffs = [0., 0.01, 0.02, 0.03]
    # cutoffs = [0.2]
    num_cutoffs = len(cutoffs) + 1
    num_days = np.shape(percentage_by_day)[1]
    # num_days = 1
    samples = []
    sample_mean = np.zeros((num_cutoffs, num_days))
    sample_std = np.zeros((num_cutoffs, num_days))
    for i in range(num_cutoffs):
        samples.append([])
    for i in range(num_days):
        for j in range(num_cutoffs):
            samples[j].append([])
    for j in range(num_cutoffs):
        for cluster in range(np.shape(percentage_by_day)[0]):
            if num_downstream_contacts_by_id[cluster] < 1:
                continue
            for day in range(num_days):
                if j < len(cutoffs) and percentage_by_day[cluster][day] > cutoffs[j]:
                    samples[j][day].append(eventual_infections[cluster])
                elif j >= len(cutoffs) and percentage_by_day[cluster][day] < 1e-6:
                    samples[j][day].append(eventual_infections[cluster])

    for j in range(num_cutoffs):
        for day in range(num_days):
            sample_mean[j][day] = np.mean(samples[j][day])
            sample_std[j][day] = np.std(samples[j][day])
    import matplotlib.pyplot as plt
    for i in range(num_cutoffs):
        plt.plot(range(num_days), sample_mean[i])
    plt.xlabel('Day of observation')
    plt.ylabel('Percentage of cluster eventually infected')
    plt.show()
    print(aggregated_infections)

def calibrate(num_index_cases):
    def get_fit(params):
        g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US, frac_SS=params[0], initial=False)
        g0_contacts = draw_contact_generation(g0_cases)
        g1_cases = draw_traced_generation_from_contacts(g0_cases, g0_contacts,
                                                        household_SAR=params[1],
                                                        external_SAR=params[2],
                                                        SS_mult=1 + np.log(1 + np.exp(params[3])),
                                                        phys_mult=1 + np.log(1 + np.exp(params[4])))
        aggregated_infections = aggregate_infections(g1_cases['id_original_case'], num_index_cases, np.sum(g1_cases['I_COVID']))
        downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
        downstream_COVID_sorted = np.flip(np.sort(downstream_COVID))
        top_10 = np.cumsum(downstream_COVID_sorted)[int(len(downstream_COVID) * 0.1)]
        R0 = np.mean(downstream_COVID)
        SS_share = top_10 / np.sum(downstream_COVID)
        print(R0, SS_share, g1_cases['household_SAR'], g1_cases['external_SAR'])
        loss =  (R0 - 2.5) ** 2 + (SS_share - 0.8) ** 2 + (g1_cases['household_SAR'] - 0.2) ** 2 + (g1_cases['external_SAR'] - 0.06) ** 2
        print(loss)
        print(params)
        print('\n')
        return loss
    res = scipy.optimize.minimize(get_fit, [0.094, 0.045, 0.012, 30, -0.28], method='Nelder-Mead')
    print(res.x)


""" no tracing, no testing """
def simulate_baseline(num_index_cases, initial=True):
    g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US, initial=initial)
    g0_contacts = draw_contact_generation(g0_cases)
    g1_cases = draw_traced_generation_from_contacts(g0_cases, g0_contacts)
    n_cases_g1 = np.sum(g1_cases['I_COVID'])
    aggregated_infections = aggregate_infections(g1_cases['id_original_case'], num_index_cases, np.sum(g1_cases['I_COVID']))
    downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
    downstream_COVID_sorted = np.flip(np.sort(downstream_COVID))
    top_10 = np.cumsum(downstream_COVID_sorted)[int(len(downstream_COVID) * 0.1)]
    print("SS share %f" % (top_10 / np.sum(downstream_COVID)))
    downstream_asymptoms = aggregated_infections[(g0_cases['I_asymptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_symptoms = aggregated_infections[(g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_COVID']).astype(bool)]
    downstream_superspreader_symptoms = aggregated_infections[(g0_cases['I_superspreader'] * g0_cases['I_symptoms'] * g0_cases['I_COVID']).astype(bool)]
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    r0 = np.mean(downstream_COVID)
    nbinom_fit = fit_dist.fit_neg_binom(downstream_COVID)
    k = nbinom_fit[1][1]
    print("frac inf by non-symptoms: %f" % (float(np.sum(g1_cases['I_infected_by_non_sympt'])) / n_cases_g1))

    plt.hist(downstream_COVID, bins=np.arange(0, downstream_COVID.max() + 1.5) - 0.5, density=True)
    plt.xlabel(r'$R_0$ (mean: %f, dispersion: %f)' % (r0, k))
    plt.ylabel(r'density')
    plt.title(r'baseline')
    plt.show()
    #
    # plt.hist(downstream_asymptoms, bins=np.arange(0, downstream_asymptoms.max() + 1.5) - 0.5, density=True)
    # plt.xlabel(r'$R_0$ if asymptoms in $g_0$ (mean: %f)' % np.mean(downstream_asymptoms))
    # plt.ylabel(r'density')
    # plt.title(r'baseline')
    # plt.show()
    #
    plt.hist(downstream_symptoms, bins=np.arange(0, downstream_symptoms.max() + 1.5) - 0.5, density=True)
    plt.xlabel(r'$R_0$ if symptoms in $g_0$ (mean: %f)' % np.mean(downstream_symptoms))
    plt.ylabel(r'density')
    plt.title(r'baseline')
    plt.show()

    plt.hist(downstream_superspreader, bins=np.arange(0, downstream_superspreader.max() + 1.5) - 0.5, density=True)
    plt.xlabel(r'$R_0$ if superspreader (mean: %f)' % np.mean(downstream_superspreader))
    plt.ylabel(r'density')
    plt.title(r'baseline')
    plt.show()

    plt.hist(downstream_superspreader_symptoms, bins=np.arange(0, downstream_superspreader_symptoms.max() + 1.5) - 0.5, density=True)
    plt.xlabel(r'$R_0$ if superspreader and symptoms (mean: %f)' % np.mean(downstream_superspreader_symptoms))
    plt.ylabel(r'density')
    plt.title(r'baseline')
    plt.show()

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

""" contact tracing, quarantine all contacts immediately on index case self-isolation """
def contact_tracing(num_index_cases, trace=False,
                    trace_superspreader=False,
                    monitor_non_SS=False,
                    bidirectional_SS=False,
                    trace_delay=0,
                    test_delay=0,
                    trace_false_negative=0.0,
                    cases_contacted=1.0,
                    ttq=False,
                    base_reduction=0.0,
                    early_release=False,
                    early_release_day=None,
                    early_release_threshold=0.,
                    wait_before_testing=0,
                    ttq_double=False,
                    monitor=False):
    g0_cases = draw_seed_index_cases(num_index_cases, age_vector_US, cases_contacted, initial=True)
    n_cases_g0 = np.sum(g0_cases['I_COVID'])
    n_contacted = np.sum(g0_cases['I_contacted'])
    g0_contacts = draw_contact_generation(g0_cases, base_reduction=base_reduction)
    print("positive tests: %i" % np.sum(g0_cases['I_test_positive']))
    traces_per_positive = np.sum([len(g0_contacts[i]) for i in np.where(g0_cases['I_contacted'])[0]]) / n_contacted
    print("avg traces per positive: %f" % traces_per_positive)

    g1_cases = draw_traced_generation_from_contacts(g0_cases=g0_cases, contacts=g0_contacts,
                                                    trace=trace, trace_superspreader=trace_superspreader,
                                                    trace_delay=trace_delay, test_delay=test_delay,
                                                    trace_false_negative=trace_false_negative,
                                                    ttq=ttq, early_release=early_release, early_release_day=early_release_day,
                                                    early_release_threshold=early_release_threshold,
                                                    wait_before_testing=wait_before_testing,
                                                    ttq_double=ttq_double,
                                                    monitor=monitor)
    num_g0_exposures = np.sum([int(np.sum([y[2] for y in x])) for x in list(g0_contacts)]) + np.sum([int(np.sum([not y[2] for y in x])) for x in list(g0_contacts)])
    num_g1_cases = np.sum(g1_cases['I_COVID'])
    print('num exposures: %i household, %i external' % (np.sum([int(np.sum([y[2] for y in x])) for x in list(g0_contacts)]),
                                                        np.sum([int(np.sum([not y[2] for y in x])) for x in list(g0_contacts)])))
    print('new index cases: ' + str(num_g1_cases))
    g1_contacts = draw_contact_generation(g1_cases, base_reduction=base_reduction)
    g2_cases = draw_traced_generation_from_contacts(g1_cases, g1_contacts, trace, trace_delay)
    aggregated_infections = np.zeros(num_index_cases)
    num_g2_cases = np.sum(g2_cases['I_COVID'])
    for i in range(int(num_g2_cases)):
        aggregated_infections[int(g2_cases['id_original_case'][i])] += 1
    downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
    r0 = np.mean(downstream_COVID)
    # nbinom_fit = fit_dist.fit_neg_binom(downstream_COVID)
    # k = nbinom_fit[1][1]
    k = -1.
    quarantine_days = list(g1_cases['n_quarantine_days'])
    tests = list(g1_cases['n_tests'])
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

    n_total_tests = np.sum(tests) + g1_cases['n_tests_of_uninfected']
    n_total_tracing_hours = n_contacted * 0.5 + 0.25 * traces_per_positive * n_contacted + (g1_cases['n_monitoring_days_of_uninfected'] + np.sum(g1_cases['n_monitoring_days']) + g1_cases['n_quarantine_days_of_uninfected'] + np.sum(g1_cases['n_quarantine_days'])) * 1/12.
    contact_tracer_rate = 20
    test_cost = 200
    print("tests")
    print(np.sum(g1_cases['n_tests']))
    print("tests of uninfected")
    print(g1_cases['n_tests_of_uninfected'])
    print("monitoring days")
    print(np.sum(g1_cases['n_monitoring_days']))
    print("monitoring days of uninfected")
    print(g1_cases['n_monitoring_days_of_uninfected'])
    print("quarantine days")
    print(np.sum(g1_cases['n_quarantine_days']))
    print("quarantine days of uninfected")
    print(g1_cases['n_quarantine_days_of_uninfected'])
    return ((r0, k),
            (np.sum(quarantine_days) + g1_cases['n_quarantine_days_of_uninfected']) / n_contacted,
            n_total_tests / n_contacted,
            (n_total_tests * test_cost + n_total_tracing_hours * contact_tracer_rate) / n_contacted,
            1 - (num_g2_cases / num_g1_cases) / raw_R0)


age_to_contact_dict = get_contacts_per_age()
