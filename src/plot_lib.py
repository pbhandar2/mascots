def scatter_3d(x, y, z, output_path):
    fig = plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1, projection='3d')

    c = np.arange(len(df))/len(df)  # create some colours
