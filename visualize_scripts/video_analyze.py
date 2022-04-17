'''
    cross category
    # attribute by video and assign an instance id
    dt_all = loadandprocessjson(reault.json)
    gt_all = loadandprocessjson(valid.json)


    # iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    iouThrs = [0.5]
    for v in videos:
        dt,gt=dt[v],gt[v]

        iou = computeIOU(dt,gt)   -->[len(dt),len(gt)]

        T=len(iouThrs)
        gtm = np.zeros((T,G))
        dtm = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        for tind, t in enumerate(iouThrs):
            for dind, d in enumerate(dt):

                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gt):

                    if gtm[tind, gind] > 0 and not iscrowd[gind]:
                        continue

                    if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                        break

                    if ious[dind, gind] < iou:
                        continue

                    iou = ious[dind, gind]
                    m = gind

                if m == -1:
                    continue
                dtIg[tind, dind] = gtIg[m]
                dtm[tind, dind] = gt[m]['id']
                gtm[tind, m] = d['id']

        dtIg = np.logical_or(dtIg, dtm == 0)

        ## visualize

        # for TP  (matched)
        # 1. iou > Thrs and category √

        # for FP or FN (unmatched)
        # 1. iou>Thrs and category √ | duplicated,
        # 2. iou>Thrs and category × | cls,
        # 3. iou<Thrs and category √ | loc,lost,
        # 4. iou<Thrs and category × | loc+cls,lost+cls,loc+lost+cls

        # visualize gt
        os.mkdirs(gt)
        save_img(gt,dir_path)

        #visualize TP
        os.mkdirs(TP)
        save_img(TP,dir_path)

        #visualize FP or FN
        os.mkdirs(errors)

        if iou>Thrs and category √:
            os.mkdirs(errors/case1)
            save_img(FP or FN ,dir_path)
        elif iou>Thrs and category ×:
            os.mkdirs(errors/case2)
            save_img(FP or FN ,dir_path)
        if iou<Thrs and category √:
            os.mkdirs(errors/case3)
            save_img(FP or FN ,dir_path)
        elif iou<Thrs and category ×:
            os.mkdirs(errors/case4)
            save_img(FP or FN ,dir_path)
'''
