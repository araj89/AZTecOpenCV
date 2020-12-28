#include <stdio.h>
#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "HybridBinarizer.h"
#include "LuminanceSource.h"
#include "GenericLuminanceSource.h"
#include "DecodeHints.h"
#include "BinaryBitmap.h"
#include "ReadBarcode.h"
#include "TextUtfEncoding.h"
#include "MultiFormatReader.h"
#include "WhiteRectDetector.h"
#include "ResultPoint.h"
#include "BitMatrix.h"
#include <math.h>

#define EPS 0.01
#define PI 3.141592

using namespace cv;
using namespace std;

static string WstringToString(const wstring& wstr) {
    std::string str;
    std::mbstate_t state = {};
    const wchar_t* data = wstr.data();
    size_t len = std::wcsrtombs(nullptr, &data, 0, &state);
    if (static_cast<size_t>(-1) != len) {
        std::unique_ptr<char[]> buff(new char[len + 1]);
        len = std::wcsrtombs(buff.get(), &data, len, &state);
        if (static_cast<size_t>(-1) != len) {
            str.assign(buff.get(), len);
        }
    }
    return str;
}


// get grid width around the bull center neighbor
// if -1 : fail
static int get_grid_width3(Mat img, int &white_val, int &black_val) {
    int ksize[] = { 5, 7, 10, 15 };
    int w, h;
    int black_thres = 60, white_thres = 100;

    // width and height
    w = img.cols;
    h = img.rows;

    //define roi which belongs to barcode area
    int l = min(w, h) / 10;
    Mat roi;
    img(Rect(w / 2 - l, h / 2 - l, l * 2, l * 2)).copyTo(roi);

    // get white, black value
    double dmin, dmax;
    minMaxLoc(roi, &dmin, &dmax);

    //if (dmin > black_thres || dmax < white_thres)
    //    return -1;


    double delta = (dmax - dmin) / 3.0;
    white_val = dmax - delta;
    black_val = dmin + delta;

    //thresholding of roi
    Mat thres_img;
    threshold(roi, thres_img, (int)black_val, 255, THRESH_BINARY_INV);
    //imshow("roi", roi);

    // loop kisze
    int i;
    for (i = 0; i < 4; i++) {
        Mat dilm;
        Mat elem = getStructuringElement(MORPH_RECT,
            Size(2 * ksize[i] + 1, 2 * ksize[i]+ 1),
            Point(ksize[i], ksize[i]));
        
        dilate(thres_img, dilm, elem);
        minMaxLoc(dilm, &dmin, &dmax);

        dilm.release();
        elem.release();
        if (dmin > 254)
            break;
    }
    i = MIN(i, 3);

    thres_img.release();
    roi.release();
    return ksize[i];

}
// determine the point (pt) is inner the polygon with 4 points or not
// if inner, return true
// else return false
// IMPORTANT : assume that the 4 points are inputted as cycle order
static bool b_pt_in_poly4(Point* poly, Point pt) {
    double s_angle = 0.0;
    for (int i = 0; i < 4; i++) {
        int j = i + 1;
        if (j == 4) j = 0;

        double dx = poly[i].x - pt.x;
        double dy = poly[i].y - pt.y;
        double dist = sqrt(dx * dx + dy * dy);
        if (dist > EPS) {
            dx /= dist;
            dy /= dist;
        }

        double dx2 = poly[j].x - pt.x;
        double dy2 = poly[j].y - pt.y;
        double dist2 = sqrt(dx2 * dx2 + dy2 * dy2);
        if (dist2 > EPS) {
            dx2 /= dist2;
            dy2 /= dist2;
        }

        double angle = acos(dx * dx2 + dy * dy2);
        s_angle += abs(angle);
    }

    if (abs(s_angle - PI * 2) < EPS) return true;
    return false;
}

// get roi from minarearect
// last output : vector<Point> of roi
static Mat get_roi_rot_rect(Mat img, Point* pt, vector<Point> &pt_outs) {
    Mat res;
    img.copyTo(res);
    uchar* data = res.data;
    int w, h;

    w = img.cols;
    h = img.rows;

    int cx = (pt[0].x + pt[1].x + pt[2].x + pt[3].x) / 4.0;
    int cy = (pt[0].y + pt[1].y + pt[2].y + pt[3].y) / 4.0;
    int r2 = (pt[0].x - cx) * (pt[0].x - cx) + (pt[0].y - cy) * (pt[0].y - cy);

    for (int i = 0; i < w * h; i++, data++) {
        int x = i % w;
        int y = i / w;

        if (*data == 0x00)
            continue;

        int cr = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (cr > r2) {
            *data = 0x00;
            continue;
        }

        if (!b_pt_in_poly4(pt, Point(x, y))) {
            *data = 0x00;
            continue;
        }


        pt_outs.push_back(Point(x, y));
    }

    return res;
}

static bool white_removed_img(Mat img, Mat& out) {
    int thres = 200;
    int l = -1, r = -1, t = -1, b = -1;
    uchar* data;
    int w, h;

    w = img.cols;
    h = img.rows;
    data = img.data;

    // get t
    for (int i = 0; i < h; i++) {
        uchar* p = data + i * w;
        bool flg = false;
        for (int j = 0; j < w; j++, p++) {
            if (*p < thres) {
                flg = true;
                break;
            }
        }
        if (flg == true) {
            t = i;
            break;
        }
    }
    if (t == -1)
        return false;

    // get b
    for (int i = h-1; i > 0; i--) {
        uchar* p = data + i * w;
        bool flg = false;
        for (int j = 0; j < w; j++, p++) {
            if (*p < thres) {
                flg = true;
                break;
            }
        }
        if (flg == true) {
            b = i;
            break;
        }
    }
    if (b == -1)
        return false;

    // get l
    for (int i = 0; i < w; i++) {
        bool flg = false;
        for (int j = 0; j < h; j++) {
            if (data[j * w + i] < thres) {
                flg = true;
                break;
            }
        }
        if (flg == true) {
            l = MAX(0, i - 1);
            break;
        }
    }
    if (l == -1)
        return false;
    
    // get r
    for (int i = w-1; i > 0; i--) {
        bool flg = false;
        for (int j = 0; j < h; j++) {
            if (data[j * w + i] < thres) {
                flg = true;
                break;
            }
        }
        if (flg == true) {
            r = MIN(w-1, i + 1);
            break;
        }
    }
    if (r == -1)
        return false;

    img(Rect(l, t, r - l, b - t)).copyTo(out);
    return true;
}

// get points from pt1 to pt2 : step = 2
static void pts_on_points(Point pt1, Point pt2, int** pxs, int** pys, int* plen) {
    double xx, yy;
    int* xs;
    int* ys;
    double step = 2.0;
    double ss;

    ss = step;

    xs = *pxs;
    ys = *pys;

    xx = pt2.x - pt1.x;
    yy = pt2.y - pt1.y;

    if (abs(xx) > abs(yy)) {
        double dy = yy / xx;
        double curx = pt1.x;
        double cury = pt1.y;
        if (xx < 0)
            step = -2.0;

        int len = 0;
        for (;;) {
            xs[len] = curx + 0.5;
            ys[len] = cury + 0.5;
            len++;
            curx += step;
            cury += step * dy;
            if (abs(curx - pt2.x) <= ss)
                break;
        }
        *plen = len;
    }
    else {
        double dx = xx / yy;
        double curx = pt1.x;
        double cury = pt1.y;
        if (yy < 0)
            step = -2.0;

        int len = 0;
        for (;;) {
            xs[len] = curx + 0.5;
            ys[len] = cury + 0.5;
            len++;
            cury += step;
            curx += step * dx;
            if (abs(cury - pt2.y) <= ss)
                break;
        }
        *plen = len;
    }
}

static bool line_on_img(Mat thres_img, Point pt1, Point pt2) {
    int w, h;
    uchar* data;
    int xs[2000];
    int ys[2000];
    int len;
    int* px;
    int* py;
    int thres;

    px = xs;
    py = ys;

    w = thres_img.cols;
    h = thres_img.rows;
    data = thres_img.data;

    pts_on_points(pt1, pt2, &px, &py, &len);
    thres = len / 30;

    int counter = 0;
    for (int i = 0; i < len; i++) {
        int v = data[xs[i] + ys[i] * w];
        if (v == 255)
            counter++;
    }
    if (counter >= thres)
        return true;
    return false;
}
// get 4 points tangent to img from img and minarearect points
static void get_adjacent_pts(Mat thres_img, Point* opts) {
    int xs[2000];
    int ys[2000];
    int len;
    int* px;
    int* py;

    px = xs;
    py = ys;

    for (int i = 0; i < 4; i++) {
        Point pt1 = opts[(i - 1 + 4) % 4];
        Point pt2 = opts[i];
        Point pt3 = opts[(i + 1) % 4];

        //  pt3      pt2
        //
        //           pt1
        pts_on_points(pt2, pt1, &px, &py, &len);

        int j;
        for (j = 0; j < len; j++) {
            if (line_on_img(thres_img, pt3, Point(xs[j], ys[j])))
                break;
        }

        // update pt2
        pt2 = Point(xs[j], ys[j]);

        pts_on_points(pt2, pt3, &px, &py, &len);
        for (j = 0; j < len; j++) {
            if (line_on_img(thres_img, pt1, Point(xs[j], ys[j])))
                break;
        }
        
        //update pt2 again
        pt2 = Point(xs[j], ys[j]);

        //update opt
        opts[i] = pt2;
    }
}

// swap if greater
static void swap_greater(int& a, int& b) {
    int c;
    if (a > b) {
        c = a;
        a = b;
        b = c;
    }
}

// get intersection point
static bool get_intersection(Point o1, Point p1, Point o2, Point p2, Point2f &r) {
    Point x = o2 - o1;
    Point d1 = p1 - o1;
    Point d2 = p2 - o2;

    float cross = d1.x * d2.y - d1.y * d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
    r = o1 + d1 * t1;
    //r.x = o1.x + (int)(d1.x * t1 + 0.5);
    //r.y = o1.y + (int)(d1.y * t1 + 0.5);
    return true;
}

static bool preprocessing2(Mat img, Mat& res) {
    Mat gimg;
    Mat thres_img, thres_img2;
    Mat warp_mat;
    Mat gimg2;
    Mat dilated;
    Mat element;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point> tCnts;
    int black_val, white_val;
    int w, h;

    w = img.cols;
    h = img.rows;

    // resize large images
    if (w > 1024) {
        h = (float)h * 1024 / w + 0.5f;
        w = 1024;
        resize(img, img, Size(w, h));
    }
    
    
    cvtColor(img, gimg, COLOR_BGR2GRAY);


    //adaptiveThreshold(gimg, gimg2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 2);
    //imshow("adaptive thres", gimg2);
    //waitKey(0);

    //if (!white_removed_img(gimg, gimg2)) {
    //    printf("image is white");
    //    gimg.release();
    //    thres_img.release();
    //    dilated.release();
    //    element.release();
    //    gimg2.release();
    //    warp_mat.release();
    //    return false;
    //}

    gimg.copyTo(gimg2);

    w = gimg.cols;
    h = gimg.rows;

    int erosion_size = get_grid_width3(gimg, white_val, black_val);

    if (erosion_size < 0) {
        printf("invalid bull center");
        gimg.release();
        thres_img.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }


    int thres = black_val;

    // get black blobs from source image
    
    medianBlur(gimg, gimg, 3);
    threshold(gimg, thres_img, thres, 255, THRESH_BINARY_INV);
    
    //erode black areas
    element = getStructuringElement(MORPH_RECT,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size));
    dilate(thres_img, dilated, element);
    
    //find blobs
    findContours(dilated, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // get the contour that has the nearest center distance from bull center
    int sel_idx = -1;
    int dist = 9999999;
    int cx = w / 2, cy = h / 2;
    for (int i = 0; i < contours.size(); i++) {
        Rect rect = boundingRect(contours[i]);
        int x = rect.x + rect.width / 2;
        int y = rect.y + rect.height / 2;
        int d = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (dist > d) {
            dist = d;
            sel_idx = i;
        }
    }

    // programming error
    if (sel_idx == -1) {
        printf("programming error");
        gimg.release();
        thres_img.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }

    // getting minarearect of the selected contours
    RotatedRect rRect = minAreaRect(Mat(contours[sel_idx]));
    Point2f vertixes[4];
    rRect.points(vertixes);



    // get barcode area
    Point verts[4];
    vector<Point>bar_pts;

    for (int i = 0; i < 4; i++) {
        verts[i].x = vertixes[i].x + 0.5;
        verts[i].y = vertixes[i].y + 0.5;
    }


    thres_img2 = get_roi_rot_rect(thres_img, verts, bar_pts);
    imshow("thres", thres_img);
    imshow("thres2", thres_img2);
    // convex hull
    vector<Point> hull;
    convexHull(bar_pts, hull);
    vector<vector<Point>>tmps, tmps2;
    tmps.push_back(hull);
    tmps2.push_back(bar_pts);
    drawContours(img, tmps, 0, Scalar(0, 255, 0), 1);

    // get four long lines points
    int id1, id2, id3, id4;
    id1 = id2 = id3 = id4 = -1;

    int l1, l2, l3, l4;
    int N = hull.size();
    if (N < 4) {
        printf("area detection fails");
        gimg.release();
        thres_img.release();
        thres_img2.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }

    l1 = l2 = l3 = l4 = -1;
    for (int i = 0; i < N; i++) {   
        int j = (i + 1) % N;
        int l = (hull[i].x - hull[j].x) * (hull[i].x - hull[j].x) + (hull[i].y - hull[j].y) * (hull[i].y - hull[j].y);

        if (l > l1) {
            // change to prev values
            l4 = l3;
            id4 = id3;
            l3 = l2;
            id3 = id2;
            l2 = l1;
            id2 = id1;
            // change cur val
            l1 = l;
            id1 = i;
        }
        else if (l > l2) {
            // change to prev values
            l4 = l3;
            id4 = id3;
            l3 = l2;
            id3 = id2;
            // change cur val
            l2 = l;
            id2 = i;
        }
        else if (l > l3) {
            // change to prev values
            id4 = id3;
            l4 = l3;
            // change cur val
            l3 = l;
            id3 = i;
        }
        else if (l > l4) {
            l4 = l;
            id4 = i;
        }
    }

    // order in ascendent
    swap_greater(id1, id2);
    swap_greater(id1, id3);
    swap_greater(id1, id4);
    swap_greater(id2, id3);
    swap_greater(id2, id4);
    swap_greater(id3, id4);

    imshow("barcode area", img);
    //imshow("dilated", dilated);
    
    bool bres;
    bres = get_intersection(hull[id1], hull[(id1 + 1) % N], hull[id2], hull[(id2 + 1) % N], vertixes[0]);
    if (!bres) {
        printf("convex hull intersection fail");
        gimg.release();
        thres_img.release();
        thres_img2.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }
    bres = get_intersection(hull[id2], hull[(id2 + 1) % N], hull[id3], hull[(id3 + 1) % N], vertixes[1]);
    if (!bres) {
        printf("convex hull intersection fail");
        gimg.release();
        thres_img.release();
        thres_img2.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }
    bres = get_intersection(hull[id3], hull[(id3 + 1) % N], hull[id4], hull[(id4 + 1) % N], vertixes[2]);
    if (!bres) {
        printf("convex hull intersection fail");
        gimg.release();
        thres_img.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }
    bres = get_intersection(hull[id4], hull[(id4 + 1) % N], hull[id1], hull[(id1 + 1) % N], vertixes[3]);
    if (!bres) {
        printf("convex hull intersection fail");
        gimg.release();
        thres_img.release();
        thres_img2.release();
        dilated.release();
        element.release();
        gimg2.release();
        warp_mat.release();
        return false;
    }

    for (int i = 0; i < 4; i++) {
        line(img, vertixes[i], vertixes[(i + 1) % 4], Scalar(0, 0, 255), 1);
    }

    // rotate 
    int s = 500;
    Point2f dst[4];
    dst[0] = Point2f(0.f, s);
    dst[1] = Point2f(0.f, 0.f);
    dst[2] = Point2f(s, 0.f);
    dst[3] = Point2f(s, s);

    warp_mat = getPerspectiveTransform(vertixes, dst);
    res = Mat::zeros(s+1, s+1, CV_8UC1);
    memset(res.data, 0xff, sizeof(uchar) * (s + 1) * (s + 1));
    warpPerspective(gimg2, res, warp_mat, res.size(), INTER_AREA);

    imshow("result", res);
    imwrite("../../imgs/res.png", res);

    gimg.release();
    thres_img.release();
    thres_img2.release();
    dilated.release();
    element.release();
    gimg2.release();
    warp_mat.release();
    return true;
}
static void Scan(Mat gray_mat) {
    

    //Width height
    int height = gray_mat.rows;
    int width = gray_mat.cols;


    //uchar* pixels;
    //pixels = new uchar[height * width];

    //memcpy(pixels, gray_mat.data, sizeof(uchar) * width * height);

    //Distinguish
    std::shared_ptr<ZXing::GenericLuminanceSource> luminance = std::make_shared<ZXing::GenericLuminanceSource>(0, 0, width, height, gray_mat.data, width * sizeof(unsigned char));
    std::shared_ptr<ZXing::BinaryBitmap> bitmap = std::make_shared<ZXing::HybridBinarizer>(luminance);

    ZXing::DecodeHints hints;
    //Add as needed format
    //std::vector<ZXing::BarcodeFormat> formats = { ZXing::BarcodeFormat(ZXing::BarcodeFormat::QR_CODE) };
    std::vector<ZXing::BarcodeFormat> formats = { ZXing::BarcodeFormat(ZXing::BarcodeFormat::AZTEC) };
    hints.setPossibleFormats(formats);
    hints = hints.setTryHarder(true);
    hints = hints.setTryRotate(true);

    auto reader = new ZXing::MultiFormatReader(hints);
    ZXing::Result result = reader->read(*bitmap);
    if (result.status() == ZXing::DecodeStatus::NoError) {
        //Recognition successful, print results
        //printf("%s", WstringToString(result.text()));
        std::string res_text = WstringToString(result.text());
        printf("%s", res_text.c_str());
    }
    else {
        printf("%s", "Failed to detect");
    }

    
    bitmap.reset();
    luminance.reset();

}

void test() {
    Mat img;
    Mat gimg;

    img = imread("../../imgs/2/10.png");

    //img = imread("../../imgs/w1_1.png");
    //img = imread("../../imgs/pics_that_dont_decode/1 (3).png");

    if (img.empty()) {
        printf("image not found\n");
        return;
    }

    typedef chrono::high_resolution_clock Time;
    typedef chrono::duration<float> fsec;

    auto t0 = Time::now();

    bool res = preprocessing2(img, gimg);
    if (!res) {
        img.release();
        gimg.release();
        printf("barcode not found\n");
        return;
    }
    Scan(gimg);

    auto t1 = Time::now();
    fsec fs = t1 - t0;

    printf("\nexecution time : %f\n", fs.count());

    waitKey(0);
    destroyAllWindows();
    img.release();
    gimg.release();
}
void main() {
    //char fpath[] = "E:\\cur_work\\LaserScanner\\CSReader_araj\\CSReader\\CalcBeadValue\\CalcBeadValue\\cat_dish.png";
    //Mat img1 = imread(fpath);
    //imshow("here", img1);
    //waitKey(0);
    //return;

    test();
    return;
    for (int i = 2; i < 57; i++) {
        char path[256];
        sprintf(path, "../../imgs/2/%d.png", (i % 57) +1);
        printf("%s", path);

        Mat img;
        Mat gimg;

        img = imread(path);

        if (img.empty()) {
            printf("image not found\n");
            continue;
        }

        typedef chrono::high_resolution_clock Time;
        typedef chrono::duration<float> fsec;

        auto t0 = Time::now();

        bool res = preprocessing2(img, gimg);
        if (!res) {
            img.release();
            gimg.release();
            printf("barcode not found\n");
            continue;
        }
        Scan(gimg);

        auto t1 = Time::now();
        fsec fs = t1 - t0;

        printf("\nexecution time : %f\n", fs.count());

        waitKey(0);
        destroyAllWindows();
        img.release();
        gimg.release();
    } 
}