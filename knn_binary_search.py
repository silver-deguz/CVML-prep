class KNN_Binary_Search:
    def __init__(self, k=1):
        self.k = k 
        self.red = None
        self.blue = None 
    
    def distance_metric(self, x, x_ref):
        return abs(x_ref - x)
    
    def train(self, a, b):
        self.red = a 
        self.blue = b
    
    def _binary_search_closest(self, x, data):
        left, right = 0, len(data)-1
        min_dist, min_idx = float('inf'), -1

        while left <= right:
            mid = left + (right - left) // 2
            dist = self.distance_metric(x, data[mid])
            dist_s = self.distance_metric(x, data[left])
            dist_e = self.distance_metric(x, data[right])

            print(dist, dist_s, dist_e)
            # print(mid, left, right)

            if dist < min_dist:
                min_dist = dist 
                min_idx = mid

            if data[mid] < x:
                right = mid-1
            elif data[mid] > x:  
                left = mid+1
            else: # data[mid] == x
                break

        print(">>>>", min_idx, data[min_idx])
        return min_idx

    def predict(self, x):
        # closest_red = self._binary_search_closest(x, self.red)
        # print(closest_red)
        closest_blue = self._binary_search_closest(x, self.blue)   
        print(closest_blue)

def main():
    red = [-0.5, 1.1, 1.4, 19, 20]
    blue = [5.5, 6, 7, 8, 9.2]
    x = 1.2

    knn_bs = KNN_Binary_Search()
    knn_bs.train(red, blue)
    knn_bs.predict(x)


if __name__ == "__main__":
    main()



