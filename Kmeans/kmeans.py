##!/usr/bin/env python

#'''
#K-means clusterization sample.
#Usage:
#   kmeans.py
#Keyboard shortcuts:
#   ESC   - exit
#   space - generate new distribution
#'''

#import numpy as np
#import cv2

#def make_gaussians(cluster_n, img_size):
#    points = []
#    ref_distrs = []
#    for i in xrange(cluster_n):
#        mean = (0.1 + 0.8*np.random.rand(2)) * img_size
#        a = (np.random.rand(2, 2)-0.5)*img_size*0.1
#        cov = np.dot(a.T, a) + img_size*0.05*np.eye(2)
#        n = 100 + np.random.randint(900)
#        pts = np.random.multivariate_normal(mean, cov, n)
#        points.append( pts )
#        ref_distrs.append( (mean, cov) )
#    points = np.float32( np.vstack(points) )
#    return points, ref_distrs

#def draw_gaussain(img, mean, cov, color):
#    x, y = np.int32(mean)
#    w, u, vt = cv2.SVDecomp(cov)
#    ang = np.arctan2(u[1, 0], u[0, 0])*(180/np.pi)
#    s1, s2 = np.sqrt(w)*3.0
#    cv2.ellipse(img, (x, y), (s1, s2), ang, 0, 360, color, 1, cv2.LINE_AA)

#if __name__ == '__main__':
#    cluster_n = 5
#    img_size = 512

#    print __doc__

#    # generating bright palette
#    colors = np.zeros((1, cluster_n, 3), np.uint8)
#    colors[0,:] = 255
#    colors[0,:,0] = np.arange(0, 180, 180.0/cluster_n)
#    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0]

#    while True:
#        print 'sampling distributions...'
#        points, _ = make_gaussians(cluster_n, img_size)

#        term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
#        ret, labels, centers = cv2.kmeans(points, cluster_n, term_crit, 10, 0)

#        img = np.zeros((img_size, img_size, 3), np.uint8)
#        for (x, y), label in zip(np.int32(points), labels.ravel()):
#            c = map(int, colors[label])
#            cv2.circle(img, (x, y), 1, c, -1)

#        cv2.imshow('gaussian mixture', img)
#        ch = 0xFF & cv2.waitKey(0)
#        if ch == 27:
#            break
#    cv2.destroyAllWindows()

import cv2
import numpy as np

img = cv2.imread('../data/15.png')
cv2.imshow('original',img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Z = gray_img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
K = 5
ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_PP_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((gray_img.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
