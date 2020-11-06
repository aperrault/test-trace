import build_contact_trees
import numpy as np
import code
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize

plt.rcParams["figure.figsize"] = (3.4, 3.4)

def run_suite(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0, dropout=0.05):
    table_mat = np.zeros((14, 5))
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=False,
                        trace_delay=trace_delay,
                        test_delay=test_delay,
                        trace_false_negative=trace_false_negative,
                        cases_contacted=cases_contacted,
                        ttq=False,
                        base_reduction=base_reduction,
                        wait_before_testing=0,
                        ttq_double=False,
                        monitor=False,
                        seed=seed)
    i = 0
    base_reff = tmp[7]
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)

    print("\nQuarantine-only")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                        trace_delay=trace_delay,
                        test_delay=test_delay,
                        trace_false_negative=trace_false_negative,
                        cases_contacted=cases_contacted,
                        ttq=False,
                        base_reduction=base_reduction,
                        wait_before_testing=0,
                        ttq_double=False,
                        monitor=False,
                        seed=seed,
                        quarantine_dropout_rate=dropout)
    i = 1
    table_mat[i][0] = np.around(tmp[7], 2)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)

    print("\nSingle Test")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=True,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 2
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    # # print("\nSingle Test Monitor")
    # # print(build_contact_trees.contact_tracing(num_index_cases, trace=True,
    # #                       trace_delay=trace_delay,
    # #                       test_delay=test_delay,
    # #                       trace_false_negative=trace_false_negative,
    # #                       cases_contacted=cases_contacted,
    # #                       ttq=True,
    # #                       base_reduction=base_reduction,
    # #                       wait_before_testing=0,
    # #                       ttq_double=False,
    # #                       monitor=True,
    # #                       seed=seed))
    print("\nDouble Test")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=True,
                          monitor=False,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 3
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)

    # # print("\nDouble Test Monitor")
    # # print(build_contact_trees.contact_tracing(num_index_cases, trace=True,
    # #                       trace_delay=trace_delay,
    # #                       test_delay=test_delay,
    # #                       trace_false_negative=trace_false_negative,
    # #                       cases_contacted=cases_contacted,
    # #                       ttq=False,
    # #                       base_reduction=base_reduction,
    # #                       wait_before_testing=0,
    # #                       ttq_double=True,
    # #                       monitor=True,
    # #                       seed=seed))
    print("\nEarly Release")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          early_release=True,
                          early_release_day=0,
                          early_release_threshold=1.e-6,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 4
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    print("\nEarly Release monitor")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          early_release=True,
                          early_release_day=0,
                          early_release_threshold=1.e-6,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=True,
                          seed=seed)
    i = 10
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)


    print("\nEarly Release and test")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=True,
                          base_reduction=base_reduction,
                          early_release=True,
                          early_release_day=0,
                          early_release_threshold=1.e-6,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 5
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    #
    # print("\nEarly Release and test monitor")
    # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
    #                       trace_delay=trace_delay,
    #                       test_delay=test_delay,
    #                       trace_false_negative=trace_false_negative,
    #                       cases_contacted=cases_contacted,
    #                       ttq=True,
    #                       base_reduction=base_reduction,
    #                       early_release=True,
    #                       early_release_day=0,
    #                       early_release_threshold=1.e-6,
    #                       wait_before_testing=0,
    #                       ttq_double=False,
    #                       monitor=True,
    #                       seed=seed)
    # i = 11
    # table_mat[i][0] = np.around(tmp[7], 3)
    # table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    # table_mat[i][2] = np.around(tmp[2][0], 1)
    # table_mat[i][3] = np.around(tmp[6][0], 1)
    # table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    print("\nEarly Release 8-4")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          early_release=True,
                          early_release_day=0,
                          early_release_threshold=1.e-6,
                          small_group_threshold=8,
                          small_group_extra=4,
                          small_group_delay=True,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 6
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    # print("\nEarly Release 8-4 test")
    # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
    #                       trace_delay=trace_delay,
    #                       test_delay=test_delay,
    #                       trace_false_negative=trace_false_negative,
    #                       cases_contacted=cases_contacted,
    #                       ttq=True,
    #                       base_reduction=base_reduction,
    #                       early_release=True,
    #                       early_release_day=0,
    #                       early_release_threshold=1.e-6,
    #                       small_group_threshold=8,
    #                       small_group_extra=4,
    #                       small_group_delay=True,
    #                       wait_before_testing=0,
    #                       ttq_double=False,
    #                       monitor=False,
    #                       seed=seed,
    #                       quarantine_dropout_rate=dropout)
    # i = 7
    # table_mat[i][0] = np.around(tmp[7], 3)
    # table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    # table_mat[i][2] = np.around(tmp[2][0], 1)
    # table_mat[i][3] = np.around(tmp[6][0], 1)
    # table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    # print("\nEarly Release hold hh")
    # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
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
    #                       monitor=False,
    #                       seed=seed,
    #                       quarantine_dropout_rate=dropout,
    #                       hold_hh=True)
    # i = 8
    # table_mat[i][0] = np.around(tmp[7], 3)
    # table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    # table_mat[i][2] = np.around(tmp[2][0], 1)
    # table_mat[i][3] = np.around(tmp[6][0], 1)
    # table_mat[i][4] = np.around(tmp[5][0], 0)
    #
    # print("\nEarly Release hold hh (exit testing)")
    # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
    #                       trace_delay=trace_delay,
    #                       test_delay=test_delay,
    #                       trace_false_negative=trace_false_negative,
    #                       cases_contacted=cases_contacted,
    #                       ttq=True,
    #                       base_reduction=base_reduction,
    #                       early_release=True,
    #                       early_release_day=0,
    #                       early_release_threshold=1.e-6,
    #                       wait_before_testing=0,
    #                       ttq_double=False,
    #                       monitor=False,
    #                       seed=seed,
    #                       quarantine_dropout_rate=dropout,
    #                       hold_hh=True)
    # i = 9
    # table_mat[i][0] = np.around(tmp[7], 3)
    # table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    # table_mat[i][2] = np.around(tmp[2][0], 1)
    # table_mat[i][3] = np.around(tmp[6][0], 1)
    # table_mat[i][4] = np.around(tmp[5][0], 0)

    # print("\nEarly Release 8-4 monitor")
    # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
    #                       trace_delay=trace_delay,
    #                       test_delay=test_delay,
    #                       trace_false_negative=trace_false_negative,
    #                       cases_contacted=cases_contacted,
    #                       ttq=False,
    #                       base_reduction=base_reduction,
    #                       early_release=True,
    #                       early_release_day=0,
    #                       early_release_threshold=1.e-6,
    #                       small_group_threshold=8,
    #                       small_group_extra=4,
    #                       small_group_delay=True,
    #                       wait_before_testing=0,
    #                       ttq_double=False,
    #                       monitor=True,
    #                       seed=seed,
    #                       quarantine_dropout_rate=dropout)
    # i = 12
    # table_mat[i][0] = np.around(tmp[7], 3)
    # table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    # table_mat[i][2] = np.around(tmp[2][0], 1)
    # table_mat[i][3] = np.around(tmp[6][0], 1)
    # table_mat[i][4] = np.around(tmp[5][0], 0)

    print("\nEarly Release 8-4 test monitor")
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=True,
                          base_reduction=base_reduction,
                          early_release=True,
                          early_release_day=0,
                          early_release_threshold=1.e-6,
                          small_group_threshold=8,
                          small_group_extra=4,
                          small_group_delay=True,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=True,
                          seed=seed,
                          quarantine_dropout_rate=dropout)
    i = 13
    table_mat[i][0] = np.around(tmp[7], 3)
    table_mat[i][1] = np.around((1 - tmp[7] / base_reff) * 100, 1)
    table_mat[i][2] = np.around(tmp[2][0], 1)
    table_mat[i][3] = np.around(tmp[6][0], 1)
    table_mat[i][4] = np.around(tmp[5][0], 0)

    print("No contact tracing & " + " & ".join(map(str, table_mat[0])) + "\\\\")
    print("Quarantine-only & " + " & ".join(map(str, table_mat[1])) + "\\\\")
    print("RBQ & " + " & ".join(map(str, table_mat[4])) + "\\\\")
    print("RBQ + exit testing & " + " & ".join(map(str, table_mat[5])) + "\\\\")
    print("RBQ + 4 extra observation days for clusters of size 8 or less & " + " & ".join(map(str, table_mat[6])) + "\\\\")
    # print("Surveillance-based release with 4 extra observation days for clusters of size 4 or less and exit testing & " + " & ".join(map(str, table_mat[7])) + "\\\\")
    # print("Surveillance-based release (non-household) & " + " & ".join(map(str, table_mat[8])) + "\\\\")
    # print("Surveillance-based release with exit testing (non-household)& " + " & ".join(map(str, table_mat[9])) + "\\\\")
    print("RBQ + active monitoring & " + " & ".join(map(str, table_mat[10])) + "\\\\")
    # print("Surveillance-based release with exit testing and monitoring& " + " & ".join(map(str, table_mat[11])) + "\\\\")
    # print("Surveillance-based release with 4 extra observation days for clusters of size 4 or less (w/ monitoring)& " + " & ".join(map(str, table_mat[12])) + "\\\\")
    print("RBQ + exit testing + 4 extra observation days for clusters of size 8 or less + active monitoring & " + " & ".join(map(str, table_mat[13])) + "\\\\")
    print("Single-test release & " + " & ".join(map(str, table_mat[2])) + "\\\\")
    print("Double-test release & " + " & ".join(map(str, table_mat[3])) + "\\\\")

    code.interact(local=locals())


def run_suite_delay(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0):
    longest_delay_to_test = 10
    num_methods = 8
    effectiveness_mat = np.zeros((num_methods, longest_delay_to_test + 1))
    quarantine_mat = np.zeros((num_methods, longest_delay_to_test + 1))
    for j in range(1, longest_delay_to_test + 1):
        print("DELAY: " , j)
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=False,
        #                     trace_delay=trace_delay,
        #                     test_delay=j,
        #                     trace_false_negative=trace_false_negative,
        #                     cases_contacted=cases_contacted,
        #                     ttq=False,
        #                     base_reduction=base_reduction,
        #                     wait_before_testing=0,
        #                     ttq_double=False,
        #                     monitor=False,
        #                     seed=seed)
        # effectiveness_mat[0][j] = tmp[7]
        # quarantine_mat[0][j] = tmp[2][0]
        print("\nQuarantine-only")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                            trace_delay=trace_delay,
                            test_delay=j,
                            trace_false_negative=trace_false_negative,
                            cases_contacted=cases_contacted,
                            ttq=False,
                            base_reduction=base_reduction,
                            wait_before_testing=0,
                            ttq_double=False,
                            monitor=False,
                            seed=seed)
        effectiveness_mat[1][j] = tmp[7]
        quarantine_mat[1][j] = tmp[2][0]
        print("\nSingle Test")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=j,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=True,
                              base_reduction=base_reduction,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        effectiveness_mat[2][j] = tmp[7]
        quarantine_mat[2][j] = tmp[2][0]
        # print("\nSingle Test Monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[3][j] = tmp[7]
        # quarantine_mat[3][j] = tmp[2][0]
        print("\nDouble Test")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=j,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              wait_before_testing=0,
                              ttq_double=True,
                              monitor=False,
                              seed=seed)
        effectiveness_mat[4][j] = tmp[7]
        quarantine_mat[4][j] = tmp[2][0]
        print(tmp)
        # print("\nDouble Test Monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=False,
        #                       base_reduction=base_reduction,
        #                       wait_before_testing=0,
        #                       ttq_double=True,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[5][j] = tmp[7]
        # quarantine_mat[5][j] = tmp[2][0]
        # print("\nEarly Release")
        # print(build_contact_trees.contact_tracing(num_index_cases, trace=True,
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
        #                       monitor=False,
        #                       seed=seed))
        print("\nEarly Release")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=j,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        effectiveness_mat[6][j] = tmp[7]
        quarantine_mat[6][j] = tmp[2][0]
        print(tmp)
        # print("\nEarly Release and test")
        # print(build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=test_delay,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       early_release=True,
        #                       early_release_day=0,
        #                       early_release_threshold=1.e-6,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=False,
        #                       seed=seed))
        # print("\nEarly Release and test monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       early_release=True,
        #                       early_release_day=0,
        #                       early_release_threshold=1.e-6,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[7][j] = tmp[7]
        # quarantine_mat[7][j] = tmp[2][0]
    np.savetxt('effectiveness.csv', effectiveness_mat)
    np.savetxt('quarantine.csv', quarantine_mat)
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), effectiveness_mat[1][1:], label="Quarantine-only")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), effectiveness_mat[2][1:], label="Single-test")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), effectiveness_mat[4][1:], label="Double-test")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), effectiveness_mat[6][1:], label="RBQ")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Time from test adminstration to results")
    plt.ylabel(r'$R_{\mathrm{eff}}$')
    plt.show()
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), quarantine_mat[1][1:], label="Quarantine-only")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), quarantine_mat[2][1:], label="Single-test")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), quarantine_mat[4][1:], label="Double-test")
    plt.plot(np.array(range(1, longest_delay_to_test + 1)), quarantine_mat[6][1:], label="RBQ")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Time from test adminstration to results")
    plt.ylabel("Quarantine days per index case")
    plt.show()
    code.interact(local=locals())

def run_suite_dropout(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0):
    largest_dropout_chance_to_test = 10
    num_methods = 8
    effectiveness_mat = np.zeros((num_methods, largest_dropout_chance_to_test + 1))
    quarantine_mat = np.zeros((num_methods, largest_dropout_chance_to_test + 1))
    for j in range(largest_dropout_chance_to_test + 1):
        base_dropout_chance = j * 0.01 * 2
        print("dropout: " , base_dropout_chance)
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=False,
        #                     trace_delay=trace_delay,
        #                     test_delay=j,
        #                     trace_false_negative=trace_false_negative,
        #                     cases_contacted=cases_contacted,
        #                     ttq=False,
        #                     base_reduction=base_reduction,
        #                     wait_before_testing=0,
        #                     ttq_double=False,
        #                     monitor=False,
        #                     seed=seed)
        # effectiveness_mat[0][j] = tmp[7]
        # quarantine_mat[0][j] = tmp[2][0]
        print("\nQuarantine-only")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                    trace_delay=trace_delay,
                    test_delay=1,
                    trace_false_negative=trace_false_negative,
                    cases_contacted=cases_contacted,
                    ttq=False,
                    base_reduction=base_reduction,
                    wait_before_testing=0,
                    ttq_double=False,
                    monitor=False,
                    seed=seed,
                    quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[0][j] = tmp[7]
        quarantine_mat[0][j] = tmp[2][0]
        # print("\nSingle Test")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=False,
        #                       seed=seed)
        # effectiveness_mat[2][j] = tmp[7]
        # quarantine_mat[2][j] = tmp[2][0]
        # print("\nSingle Test Monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[3][j] = tmp[7]
        # quarantine_mat[3][j] = tmp[2][0]
        print("\nDouble Test")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              wait_before_testing=0,
                              ttq_double=True,
                              monitor=False,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[1][j] = tmp[7]
        quarantine_mat[1][j] = tmp[2][0]
        # print("\nDouble Test Monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=False,
        #                       base_reduction=base_reduction,
        #                       wait_before_testing=0,
        #                       ttq_double=True,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[5][j] = tmp[7]
        # quarantine_mat[5][j] = tmp[2][0]
        print("\nEarly Release")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[2][j] = tmp[7]
        quarantine_mat[2][j] = tmp[2][0]
        # print("\nEarly Release + 1")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=1,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=False,
        #                       base_reduction=base_reduction,
        #                       early_release=True,
        #                       early_release_day=1,
        #                       early_release_threshold=1.e-6,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=False,
        #                       seed=seed,
        #                       quarantine_dropout_rate=j * 0.01 * 2)
        # effectiveness_mat[4][j] = tmp[7]
        # quarantine_mat[4][j] = tmp[2][0]
        print("\nEarly Release monitor")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=True,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[3][j] = tmp[7]
        quarantine_mat[3][j] = tmp[2][0]
        print("\nEarly Release monitor + 1")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=1,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=True,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[5][j] = tmp[7]
        quarantine_mat[5][j] = tmp[2][0]
        print("\nEarly Release monitor + 2")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=2,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=True,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[6][j] = tmp[7]
        quarantine_mat[6][j] = tmp[2][0]
        print("\nEarly Release small")
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=1,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              small_group_delay=True,
                              small_group_threshold=8,
                              small_group_extra=4,
                              monitor=True,
                              seed=seed,
                              quarantine_dropout_rate=j * 0.01 * 2)
        effectiveness_mat[7][j] = tmp[7]
        quarantine_mat[7][j] = tmp[2][0]

        # print("\nEarly Release and test")
        # print(build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=test_delay,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       early_release=True,
        #                       early_release_day=0,
        #                       early_release_threshold=1.e-6,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=False,
        #                       seed=seed))
        # print("\nEarly Release and test monitor")
        # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
        #                       trace_delay=trace_delay,
        #                       test_delay=j,
        #                       trace_false_negative=trace_false_negative,
        #                       cases_contacted=cases_contacted,
        #                       ttq=True,
        #                       base_reduction=base_reduction,
        #                       early_release=True,
        #                       early_release_day=0,
        #                       early_release_threshold=1.e-6,
        #                       wait_before_testing=0,
        #                       ttq_double=False,
        #                       monitor=True,
        #                       seed=seed)
        # effectiveness_mat[7][j] = tmp[7]
        # quarantine_mat[7][j] = tmp[2][0]
    np.savetxt('effectiveness_dropout.csv', effectiveness_mat)
    np.savetxt('quarantine_dropout.csv', quarantine_mat)
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[0], label="Quarantine-only")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[1], label="Double-test")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[3], label="Surveillance-based")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[5], label="Surveillance-based + 1 day")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[6], label="Surveillance-based + 2 day")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[7], label="Surveillance-based small group")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, effectiveness_mat[2], label="Surveillance-based (no monitor)")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Base quarantine dropout chance")
    plt.ylabel(r'$R_{\mathrm{eff}}$')
    plt.show()
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[0], label="Quarantine-only")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[1], label="Double-test")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[3], label="Surveillance-based")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[5], label="Surveillance-based + 1 day")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[6], label="Surveillance-based + 2 day")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[7], label="Surveillance-based small group")
    plt.plot(np.array(range(largest_dropout_chance_to_test + 1)) * 0.01 * 2, quarantine_mat[2], label="Surveillance-based (no monitor)")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("Base quarantine dropout chance")
    plt.ylabel("Quarantine days per index case")
    plt.show()
    code.interact(local=locals())


def run_sweep_small_group(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0):
    largest_group_to_test = 10
    smallest_group_to_test = 5
    smallest_delay_to_test = 1
    largest_delay_to_test = 5
    small_group_Rs = np.zeros((largest_delay_to_test - smallest_delay_to_test + 1, largest_group_to_test - smallest_group_to_test + 1))
    small_group_Qs = np.zeros((largest_delay_to_test - smallest_delay_to_test + 1, largest_group_to_test - smallest_group_to_test + 1))
    small_group_Rs_test = np.zeros((largest_delay_to_test - smallest_delay_to_test + 1, largest_group_to_test - smallest_group_to_test + 1))
    small_group_Qs_test = np.zeros((largest_delay_to_test - smallest_delay_to_test + 1, largest_group_to_test - smallest_group_to_test + 1))
    for i in range(smallest_group_to_test, largest_group_to_test + 1):
        for j in range(smallest_delay_to_test, largest_delay_to_test + 1):
            tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                                  trace_delay=trace_delay,
                                  test_delay=test_delay,
                                  trace_false_negative=trace_false_negative,
                                  cases_contacted=cases_contacted,
                                  ttq=False,
                                  base_reduction=base_reduction,
                                  early_release=True,
                                  early_release_day=0,
                                  early_release_threshold=1.e-6,
                                  small_group_delay=True,
                                  small_group_threshold=i,
                                  small_group_extra=j,
                                  wait_before_testing=0,
                                  ttq_double=False,
                                  monitor=False,
                                  seed=seed)
            small_group_Rs[j - smallest_delay_to_test][i - smallest_group_to_test] = tmp[7]
            small_group_Qs[j - smallest_delay_to_test][i - smallest_group_to_test] = tmp[2][0]
            # tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
            #                       trace_delay=trace_delay,
            #                       test_delay=test_delay,
            #                       trace_false_negative=trace_false_negative,
            #                       cases_contacted=cases_contacted,
            #                       ttq=True,
            #                       base_reduction=base_reduction,
            #                       early_release=True,
            #                       early_release_day=0,
            #                       early_release_threshold=1.e-6,
            #                       small_group_delay=True,
            #                       small_group_threshold=i,
            #                       small_group_extra=j,
            #                       wait_before_testing=0,
            #                       ttq_double=False,
            #                       monitor=True,
            #                       seed=seed)
            # small_group_Rs_test[j - smallest_delay_to_test][i - smallest_group_to_test] = tmp[7]
            # small_group_Qs_test[j - smallest_delay_to_test][i - smallest_group_to_test] = tmp[2][0]
    print(small_group_Rs)
    print(small_group_Qs)
    code.interact(local=locals())

def run_sweep_gruber(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0):
    num_methods = 4
    smallest_group_to_test = 0
    largest_group_to_test = 30

    small_group_Rs = np.zeros((num_methods, largest_group_to_test - smallest_group_to_test + 1))
    small_group_Qs = np.zeros((num_methods, largest_group_to_test - smallest_group_to_test + 1))
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=False,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed)
    small_group_Rs[0][:] = tmp[6][0]
    small_group_Qs[0][:] = tmp[2][0]
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed)
    small_group_Rs[1][:] = tmp[6][0]
    small_group_Qs[1][:] = tmp[2][0]
    for i in range(smallest_group_to_test, largest_group_to_test + 1):
        print("GROUP SIZE ", i)
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              small_group_delay=True,
                              small_group_threshold=i,
                              small_group_extra=14,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        small_group_Rs[2][i - smallest_group_to_test] = tmp[6][0]
        small_group_Qs[2][i - smallest_group_to_test] = tmp[2][0]
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=True,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              small_group_delay=True,
                              small_group_threshold=i,
                              small_group_extra=14,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        small_group_Rs[3][i - smallest_group_to_test] = tmp[6][0]
        small_group_Qs[3][i - smallest_group_to_test] = tmp[2][0]
    print(small_group_Rs)
    print(small_group_Qs)
    code.interact(local=locals())
    xs = np.array(range(largest_group_to_test - smallest_group_to_test + 1))
    plt.plot(xs, small_group_Rs[1], label='Quarantine-only')
    plt.plot(xs, small_group_Rs[2], label='Surveillance-based')
    plt.plot(xs, small_group_Rs[3], label='Surveillance-based with testing')
    plt.xlabel("Cluster size threshold for surveillance-based release")
    plt.ylabel("Deaths per 1000 index cases")
    plt.legend()
    plt.show()
    plt.plot(xs, small_group_Qs[0], label='No tracing')
    plt.plot(xs, small_group_Qs[1], label='Quarantine-only')
    plt.plot(xs, small_group_Qs[2], label='Surveillance-based')
    plt.plot(xs, small_group_Qs[3], label='Surveillance-based with testing')
    plt.xlabel("Cluster size threshold for surveillance-based release")
    plt.ylabel("Quarantine days per index case")
    plt.legend()
    plt.show()


def run_sweep_gruber2(num_index_cases, max_clustersize, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, seed=0):
    num_methods = 4
    smallest_delay_to_test = 0
    largest_delay_to_test = 14

    small_group_Rs = np.zeros((num_methods, largest_delay_to_test - smallest_delay_to_test + 1))
    small_group_Qs = np.zeros((num_methods, largest_delay_to_test - smallest_delay_to_test + 1))
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=False,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed)
    small_group_Rs[0][:] = tmp[6][0]
    small_group_Qs[0][:] = tmp[2][0]
    tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                          trace_delay=trace_delay,
                          test_delay=test_delay,
                          trace_false_negative=trace_false_negative,
                          cases_contacted=cases_contacted,
                          ttq=False,
                          base_reduction=base_reduction,
                          wait_before_testing=0,
                          ttq_double=False,
                          monitor=False,
                          seed=seed)
    small_group_Rs[1][:] = tmp[6][0]
    small_group_Qs[1][:] = tmp[2][0]
    for i in range(smallest_delay_to_test, largest_delay_to_test + 1):
        print("DELAY SIZE ", i)
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              small_group_delay=True,
                              small_group_threshold=max_clustersize,
                              small_group_extra=i,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        small_group_Rs[2][i - smallest_delay_to_test] = tmp[6][0]
        small_group_Qs[2][i - smallest_delay_to_test] = tmp[2][0]
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=True,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=0,
                              early_release_threshold=1.e-6,
                              small_group_delay=True,
                              small_group_threshold=max_clustersize,
                              small_group_extra=i,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        small_group_Rs[3][i - smallest_delay_to_test] = tmp[6][0]
        small_group_Qs[3][i - smallest_delay_to_test] = tmp[2][0]
    print(small_group_Rs)
    print(small_group_Qs)
    code.interact(local=locals())
    xs = np.array(range(largest_delay_to_test - smallest_delay_to_test + 1))
    plt.plot(xs, small_group_Rs[1], label='Quarantine-only')
    plt.plot(xs, small_group_Rs[2], label='Surveillance-based')
    plt.plot(xs, small_group_Rs[3], label='Surveillance-based with testing')
    plt.xlabel("Additional days of observation")
    plt.ylabel("Deaths per 1000 index cases")
    plt.legend()
    plt.show()
    plt.plot(xs, small_group_Qs[0], label='No tracing')
    plt.plot(xs, small_group_Qs[1], label='Quarantine-only')
    plt.plot(xs, small_group_Qs[2], label='Surveillance-based')
    plt.plot(xs, small_group_Qs[3], label='Surveillance-based with testing')
    plt.xlabel("Additional days of observation")
    plt.ylabel("Quarantine days per index case")
    plt.legend()
    plt.show()

def run_sweep_early_release(num_index_cases, trace_delay=0, test_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, early_release_threshold=1.e-6, seed=0):
    smallest_delay_to_test = 0
    largest_delay_to_test = 5
    sweep_R = np.zeros(largest_delay_to_test - smallest_delay_to_test + 1)
    sweep_Q = np.zeros(largest_delay_to_test - smallest_delay_to_test + 1)
    for j in range(smallest_delay_to_test, largest_delay_to_test + 1):
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=j,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        sweep_R[j - smallest_delay_to_test] = tmp[0][0]
        sweep_Q[j - smallest_delay_to_test] = tmp[2][0]
    print(sweep_R)
    print(sweep_Q)

def run_sweep_test_delay(num_index_cases, trace_delay=0,
              trace_false_negative=0.0, cases_contacted=0.0,
              base_reduction=0.0, early_release_threshold=1.e-6, seed=0):
    smallest_delay_to_test = 0
    largest_delay_to_test = 5
    sweep_R = np.zeros(largest_delay_to_test - smallest_delay_to_test + 1)
    sweep_Q = np.zeros(largest_delay_to_test - smallest_delay_to_test + 1)
    for j in range(smallest_delay_to_test, largest_delay_to_test + 1):
        tmp = build_contact_trees.contact_tracing(num_index_cases, trace=True,
                              trace_delay=trace_delay,
                              test_delay=test_delay,
                              trace_false_negative=trace_false_negative,
                              cases_contacted=cases_contacted,
                              ttq=False,
                              base_reduction=base_reduction,
                              early_release=True,
                              early_release_day=j,
                              early_release_threshold=1.e-6,
                              wait_before_testing=0,
                              ttq_double=False,
                              monitor=False,
                              seed=seed)
        sweep_R[j - smallest_delay_to_test] = tmp[0][0]
        sweep_Q[j - smallest_delay_to_test] = tmp[2][0]
    print(sweep_R)
    print(sweep_Q)


def draw_samples(num_index_cases, trace_delay, test_delay, wait_before_testing,
                 trace_false_negative, cases_contacted, base_reduction=0.5, seed=0):
    g0_cases = draw_seed_index_cases(num_individuals=num_index_cases, cases_contacted=cases_contacted,
                                     age_vector=age_vector_US, initial=True)
    g0_contacts = draw_contact_generation(g0_cases, base_reduction=base_reduction, seed=seed)
    (n_age, t_exposure, t_last_exposure, original_case, infected_by, I_COVID, successful_traces,
    quarantine_days_of_uninfected, tests_of_uninfected, monitoring_days_of_uninfected,
    infected_by_presymp, infected_by_asymp, uninfected_source, uninfected_exposure,
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
        tfn=trace_false_negative,
        wait_before_testing=wait_before_testing)

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

    quarantine_by_parent_case_release = np.zeros(len(g0_cases['I_COVID']))

    uninf_false_positive = np.random.geometric(p=symptom_false_positive_chance, size=len(uninfected_source)) + uninfected_exposure
    if early_release_day is None:
        early_release_day = 0

    precalc_dropout = 0
    dropouts = np.zeros(len(g1_cases['I_symptoms']) + len(uninfected_source)).reshape(-1, 1)
    # symptom check is one day before release
    quarantine_early_release_day = early_release_day + 1
    early_release_by_parent_case = np.ones(len(g0_cases['I_COVID'])) * quarantine_early_release_day

    min_size = 1
    max_size = 6
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
    quarantine_by_parent_case_release = (percentage_by_day < 1.e-6)

    traced_COVID_by_ID = np.zeros(len(num_downstream_traces_by_id))
    for i in range(len(g1_cases['I_symptoms'])):
        if successful_traces[i]:
            parent_case = int(g1_cases['id_infected_by'][i])
            traced_COVID_by_ID[parent_case] += 1

    clusters_released_by_size = np.zeros((max_size - min_size + 1, max_day - min_day + 1))
    clusters_total_by_size = np.zeros(max_size - min_size + 1)
    COVID_released_by_size = np.zeros((max_size - min_size + 1, max_day - min_day + 1))
    COVID_total_by_size = np.zeros(max_size - min_size + 1)

    for i in range(len(num_downstream_traces_by_id)):
        size = int(num_downstream_traces_by_id[i])
        if min_size <= size <= max_size:
            clusters_total_by_size[size] += 1
            COVID_total_by_size[size] +=1
        else:
            continue
        for day in range(max_day):
            if percentage_by_day[i][day] < 1.e-6:
                COVID_released_by_size[size] += traced_COVID_by_ID[i]
    print("clusters_total_by_size", clusters_total_by_size)
    print("COVID total by size", COVID_total_by_size)

    print("clusters released by size", clusters_released_by_size)
    print("COVID total by size", COVID_released_by_size)


    # import matplotlib.pyplot as plt
    # for i in range(num_cutoffs):
    #     plt.plot(range(num_days), sample_mean[i])
    # plt.xlabel('Day of observation')
    # plt.ylabel('Percentage of cluster eventually infected')
    # plt.show()
    # print(aggregated_infections)

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

def calibrate(num_index_cases):
    def get_fit(params):
        seed = 0
        np.random.seed(seed)
        g0_cases = build_contact_trees.draw_seed_index_cases(num_index_cases, build_contact_trees.age_vector_US, frac_SS=1 / (1 + np.exp(-params[0])), initial=False)
        g0_contacts = build_contact_trees.draw_contact_generation(g0_cases, seed=seed)
        g1_cases = build_contact_trees.draw_traced_generation_from_contacts(g0_cases, g0_contacts,
                                                        household_SAR=1 / (1 + np.exp(-params[1])),
                                                        external_SAR=1 / (1 + np.exp(-params[2])),
                                                        SS_mult=1 + np.log(1 + np.exp(params[3])),
                                                        phys_mult=1 + np.log(1 + np.exp(params[4])),
                                                        seed=seed)
        aggregated_infections = build_contact_trees.aggregate_infections(g1_cases['id_original_case'], num_index_cases, np.sum(g1_cases['I_COVID']))
        downstream_COVID = (aggregated_infections[(g0_cases['I_COVID']).astype(bool)]).astype(int)
        downstream_COVID_sorted = np.flip(np.sort(downstream_COVID))
        top_10 = np.cumsum(downstream_COVID_sorted)[int(len(downstream_COVID) * 0.2)]
        R0 = np.mean(downstream_COVID)
        SS_share = top_10 / np.sum(downstream_COVID)
        print(R0, SS_share, g1_cases['household_SAR'], g1_cases['external_SAR'])
        loss = ((SS_share - 0.8)) ** 2 + ((g1_cases['household_SAR'] - 0.188) * 4) ** 2 + ((g1_cases['external_SAR'] - 0.06) * 2) ** 2
        # loss = ((R0 - 2.) / 5) ** 2 + ((SS_share - 0.75)) ** 2 + ((g1_cases['household_SAR'] - 0.188) * 4) ** 2 + ((g1_cases['external_SAR'] - 0.06) * 2) ** 2
        # loss = ((SS_share - 0.8) * 7) ** 2 + (g1_cases['household_SAR'] - 0.2) ** 2 + (g1_cases['external_SAR'] - 0.06) ** 2
        print(loss)
        print(params)
        print('\n')
        return loss
    res = scipy.optimize.minimize(get_fit, [-1.9, -3.6, -5., 26.3, 4.1], method='Nelder-Mead')
    # res = scipy.optimize.minimize(get_fit, [0.094, 0.047, 0.014, 30, -0.28], method='Nelder-Mead')
    print(res.x)

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
    print("frac inf by asymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_asymp'])) / n_cases_g1))
    print("frac inf by presymptoms: %f" % (float(np.sum(g1_cases['I_infected_by_presymp'])) / n_cases_g1))
    return ((r0, k), 0., np.sum(downstream_asymptoms) / np.sum(g0_cases['I_asymptoms']), np.sum(downstream_symptoms) / np.sum(g0_cases['I_symptoms']), 1 - (n_cases_g1 / n_cases_g0) / raw_R0)
