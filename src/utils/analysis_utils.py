def get_stratified_random_sample(df_agreement, ref_acc, n_sample = 30, tolerance=0.05, min_seed=0, max_seed=20000):
    checker = True
    for seed_acc in range(min_seed,max_seed):
        found = False
        df_sampled = df_agreement.sample(n=n_sample, random_state=seed_acc)
        acc = df_sampled.acc.mean()
        if abs(acc-ref_acc)/ref_acc < tolerance:
            print('Random seed found')
            found = True
            break
        if not found:
            checker = checker and False
    print(seed_acc, ref_acc, acc)
    return checker, seed_acc, ref_acc, acc, df_sampled