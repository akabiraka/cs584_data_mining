def extract_features_for_0(img_pixels, class_label):
    # print(img_pixels.shape)
    print("extracting features for {} ... ...", (class_label))
    points = []
    row = 0
    while row != 125:
        row = row + 25
        x_low = 0
        x_high = 100
        y_low = row-25
        y_high = row
        pre_dist = 0.0
        # my_x1, my_y1, my_x2, my_y2
        search_limit = 5000
        got_two_points = False
        i = 1
        while i!=10:
            i = i + 1
            search_count = 0
            got_first_point = False
            got_second_point = False
            while True:
                search_count = search_count + 1
                x1, y1 = np.random.randint(low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
                if img_pixels[x1, y1]:
                    got_first_point = True
                    break
                if search_count == search_limit:
                    break

            search_count = 0
            while True:
                search_count = search_count + 1
                x2, y2 = np.random.randint(low=x_low, high=x_high), np.random.randint(low=y_low, high=y_high)
                if img_pixels[x2, y2]:
                    got_second_point = True
                    break
                if search_count == search_limit:
                    break

            if(got_first_point and got_second_point):
                dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
                if dist > pre_dist:
                    pre_dist = dist
                    my_x1 = x1
                    my_y1 = y1
                    my_x2 = x2
                    my_y2 = y2
                    got_two_points = True

        if got_two_points:
            print("got two points...")
            if len(points) != 16:
                points.append(my_x1)
                points.append(my_y1)
            if len(points) != 16:
                points.append(my_x2)
                points.append(my_y2)
        else:
            print("didn't found two points in ", (y_low), (y_high))
            if y_low == 75 and y_high == 100:
                row = 0

        print(len(points), row)
        if len(points) != 16 and row == 100:
            row = 0
        elif len(points) == 16:
            break

    points.append(class_label)
    print(points)
    return points
