/*
* Copyright 2016 Nu-book Inc.
* Copyright 2016 ZXing authors
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "AZDetector.h"
#include "AZDetectorResult.h"
#include "BitHacks.h"
#include "ZXNumeric.h"
#include "ReedSolomonDecoder.h"
#include "GenericGF.h"
#include "WhiteRectDetector.h"
#include "GridSampler.h"
#include "DecodeStatus.h"
#include "BitMatrix.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <array>
#include <math.h>
using namespace cv;

namespace ZXing {
namespace Aztec {

static const int EXPECTED_CORNER_BITS[] = {
	0xee0,  // 07340  XXX .XX X.. ...
	0x1dc,  // 00734  ... XXX .XX X..
	0x83b,  // 04073  X.. ... XXX .XX
	0x707,  // 03407 .XX X.. ... XXX
};

// return -1 in case of error
static int GetRotation(const std::array<int, 4>& sides, int length)
{
	// In a normal pattern, we expect to See
	//   **    .*             D       A
	//   *      *
	//
	//   .      *
	//   ..    ..             C       B
	//
	// Grab the 3 bits from each of the sides the form the locator pattern and concatenate
	// into a 12-bit integer.  Start with the bit at A
	int cornerBits = 0;
	for (int side : sides) {
		// XX......X where X's are orientation marks
		int t = ((side >> (length - 2)) << 1) + (side & 1);
		cornerBits = (cornerBits << 3) + t;
	}
	// Mov the bottom bit to the top, so that the three bits of the locator pattern at A are
	// together.  cornerBits is now:
	//  3 orientation bits at A || 3 orientation bits at B || ... || 3 orientation bits at D
	cornerBits = ((cornerBits & 1) << 11) + (cornerBits >> 1);
	// The result shift indicates which element of BullsEyeCorners[] goes into the top-left
	// corner. Since the four rotation values have a Hamming distance of 8, we
	// can easily tolerate two errors.
	for (int shift = 0; shift < 4; shift++) {
		if (BitHacks::CountBitsSet(cornerBits ^ EXPECTED_CORNER_BITS[shift]) <= 2) {
			return shift;
		}
	}
	return -1;
}

inline static bool IsValidPoint(int x, int y, int imgWidth, int imgHeight)
{
	return x >= 0 && x < imgWidth && y > 0 && y < imgHeight;
}

inline static bool IsValidPoint(const ResultPoint& point, int imgWidth, int imgHeight)
{
	return IsValidPoint(RoundToNearest(point.x()), RoundToNearest(point.y()), imgWidth, imgHeight);
}

//private static float distance(Point a, Point b) {
//	return MathUtils.distance(a.getX(), a.getY(), b.getX(), b.getY());
//}
//
//private static float distance(ResultPoint a, ResultPoint b) {
//	return MathUtils.distance(a.getX(), a.getY(), b.getX(), b.getY());
//}

/**
* Samples a line.
*
* @param p1   start point (inclusive)
* @param p2   end point (exclusive)
* @param size number of bits
* @return the array of bits as an int (first bit is high-order bit of result)
*/
static int SampleLine(const BitMatrix& image, const ResultPoint& p1, const ResultPoint& p2, int size)
{
	int result = 0;

	float d = ResultPoint::Distance(p1, p2);
	float moduleSize = d / size;
	float px = p1.x();
	float py = p1.y();
	float dx = moduleSize * (p2.x() - p1.x()) / d;
	float dy = moduleSize * (p2.y() - p1.y()) / d;
	for (int i = 0; i < size; i++) {
		if (image.get(RoundToNearest(px + i * dx), RoundToNearest(py + i * dy))) {
			result |= 1 << (size - i - 1);
		}
	}
	return result;
}

/**
* Corrects the parameter bits using Reed-Solomon algorithm.
*
* @param parameterData parameter bits
* @param compact true if this is a compact Aztec code
*/
static bool GetCorrectedParameterData(int64_t parameterData, bool compact, int& result)
{
	int numCodewords;
	int numDataCodewords;

	if (compact) {
		numCodewords = 7;
		numDataCodewords = 2;
	}
	else {
		numCodewords = 10;
		numDataCodewords = 4;
	}

	int numECCodewords = numCodewords - numDataCodewords;
	std::vector<int> parameterWords(numCodewords);
	for (int i = numCodewords - 1; i >= 0; --i) {
		parameterWords[i] = (int)parameterData & 0xF;
		parameterData >>= 4;
	}
	if (!ReedSolomonDecoder::Decode(GenericGF::AztecParam(), parameterWords, numECCodewords))
		return false;

	// Toss the error correction.  Just return the data as an integer
	result = 0;
	for (int i = 0; i < numDataCodewords; i++) {
		result = (result << 4) + parameterWords[i];
	}
	return true;
}

/**
* Extracts the number of data layers and data blocks from the layer around the bull's eye.
*
* @param bullsEyeCorners the array of bull's eye corners
* @throws NotFoundException in case of too many errors or invalid parameters
*/
static bool ExtractParameters(const BitMatrix& image, const std::array<ResultPoint, 4>& bullsEyeCorners, bool compact, int nbCenterLayers, int& nbLayers, int& nbDataBlocks, int& shift)
{
	if (!IsValidPoint(bullsEyeCorners[0], image.width(), image.height()) || !IsValidPoint(bullsEyeCorners[1], image.width(), image.height()) ||
		!IsValidPoint(bullsEyeCorners[2], image.width(), image.height()) || !IsValidPoint(bullsEyeCorners[3], image.width(), image.height())) {
		return false;
	}
	int length = 2 * nbCenterLayers;
	// Get the bits around the bull's eye
	std::array<int, 4> sides = {
		SampleLine(image, bullsEyeCorners[0], bullsEyeCorners[1], length), // Right side
		SampleLine(image, bullsEyeCorners[1], bullsEyeCorners[2], length), // Bottom 
		SampleLine(image, bullsEyeCorners[2], bullsEyeCorners[3], length), // Left side
		SampleLine(image, bullsEyeCorners[3], bullsEyeCorners[0], length)  // Top 
	};

	// bullsEyeCorners[shift] is the corner of the bulls'eye that has three 
	// orientation marks.  
	// sides[shift] is the row/column that goes from the corner with three
	// orientation marks to the corner with two.
	shift = GetRotation(sides, length);
	if (shift < 0) {
		return false;
	}

	// Flatten the parameter bits into a single 28- or 40-bit long
	int64_t parameterData = 0;
	for (int i = 0; i < 4; i++) {
		int side = sides[(shift + i) % 4];
		if (compact) {
			// Each side of the form ..XXXXXXX. where Xs are parameter data
			parameterData <<= 7;
			parameterData += (side >> 1) & 0x7F;
		}
		else {
			// Each side of the form ..XXXXX.XXXXX. where Xs are parameter data
			parameterData <<= 10;
			parameterData += ((side >> 2) & (0x1f << 5)) + ((side >> 1) & 0x1F);
		}
	}

	// Corrects parameter data using RS.  Returns just the data portion
	// without the error correction.
	int correctedData;
	if (!GetCorrectedParameterData(parameterData, compact, correctedData)) {
		return false;
	}

	if (compact) {
		// 8 bits:  2 bits layers and 6 bits data blocks
		nbLayers = (correctedData >> 6) + 1;
		nbDataBlocks = (correctedData & 0x3F) + 1;
	}
	else {
		// 16 bits:  5 bits layers and 11 bits data blocks
		nbLayers = (correctedData >> 11) + 1;
		nbDataBlocks = (correctedData & 0x7FF) + 1;
	}
	return true;
}

struct PixelPoint
{
	int x;
	int y;

	ResultPoint toResultPoint() const { return {x, y}; }
};


inline static float Distance(const PixelPoint& a, const PixelPoint& b)
{
	return ResultPoint::Distance(a.x, a.y, b.x, b.y);
}


/**
* Gets the color of a segment
*
* @return 1 if segment more than 90% black, -1 if segment is more than 90% white, 0 else
*/
static int GetColor(const BitMatrix& image, const PixelPoint& p1, const PixelPoint& p2)
{
	if (!IsValidPoint(p1.x, p1.y, image.width(), image.height()) ||
		!IsValidPoint(p2.x, p2.y, image.width(), image.height()))
		return 0;

	float d = Distance(p1, p2);
	float dx = (p2.x - p1.x) / d;
	float dy = (p2.y - p1.y) / d;
	int error = 0;

	float px = static_cast<float>(p1.x);
	float py = static_cast<float>(p1.y);

	bool colorModel = image.get(p1.x, p1.y);
	int iMax = (int)std::ceil(d);
	for (int i = 0; i < iMax; i++) {
		px += dx;
		py += dy;
		if (image.get(RoundToNearest(px), RoundToNearest(py)) != colorModel) {
			error++;
		}
	}

	float errRatio = error / d;

	if (errRatio > 0.1f && errRatio < 0.9f) {
		return 0;
	}

	return (errRatio <= 0.1f) == colorModel ? 1 : -1;
}

/**
* @return true if the border of the rectangle passed in parameter is compound of white points only
*         or black points only
*/
static bool IsWhiteOrBlackRectangle(const BitMatrix& image, const PixelPoint& pt1, const PixelPoint& pt2, const PixelPoint& pt3, const PixelPoint& pt4) {

	//araj
	//int corr = 3;
	int corr = 2;


	PixelPoint p1{ pt1.x - corr, pt1.y + corr };
	PixelPoint p2{ pt2.x - corr, pt2.y - corr };
	PixelPoint p3{ pt3.x + corr, pt3.y - corr };
	PixelPoint p4{ pt4.x + corr, pt4.y + corr };

	int cInit = GetColor(image, p4, p1);

	if (cInit == 0) {
		return false;
	}

	int c = GetColor(image, p1, p2);

	if (c != cInit) {
		return false;
	}

	c = GetColor(image, p2, p3);

	if (c != cInit) {
		return false;
	}

	c = GetColor(image, p3, p4);

	return c == cInit;

}

/**
* Gets the coordinate of the first point with a different color in the given direction
*/
static PixelPoint GetFirstDifferent(const BitMatrix& image, const PixelPoint& init, bool color, int dx, int dy) {
	int x = init.x + dx;
	int y = init.y + dy;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get(x, y) == color) {
		x += dx;
		y += dy;
	}

	x -= dx;
	y -= dy;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get(x, y) == color) {
		x += dx;
	}
	x -= dx;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get(x, y) == color) {
		y += dy;
	}
	y -= dy;

	return PixelPoint{ x, y };
}

static PixelPoint GetFirstDifferent2(const BitMatrix& image, const PixelPoint& init, bool color, int dx, int dy) {
	int x = init.x + dx;
	int y = init.y + dy;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get3(x, y, color) == color) {
		x += dx;
		y += dy;
	}

	x -= dx;
	y -= dy;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get3(x, y, color) == color) {
		x += dx;
	}
	x -= dx;

	while (IsValidPoint(x, y, image.width(), image.height()) && image.get3(x, y, color) == color) {
		y += dy;
	}
	y -= dy;

	return PixelPoint{ x, y };
}

/**
* Expand the square represented by the corner points by pushing out equally in all directions
*
* @param cornerPoints the corners of the square, which has the bull's eye at its center
* @param oldSide the original length of the side of the square in the target bit matrix
* @param newSide the new length of the size of the square in the target bit matrix
* @return the corners of the expanded square
*/
static void ExpandSquare(std::array<ResultPoint, 4>& cornerPoints, float oldSide, float newSide)
{
	float ratio = newSide / (2 * oldSide);
	float dx = cornerPoints[0].x() - cornerPoints[2].x();
	float dy = cornerPoints[0].y() - cornerPoints[2].y();
	float centerx = (cornerPoints[0].x() + cornerPoints[2].x()) / 2.0f;
	float centery = (cornerPoints[0].y() + cornerPoints[2].y()) / 2.0f;

	cornerPoints[0] = ResultPoint(centerx + ratio * dx, centery + ratio * dy);
	cornerPoints[2] = ResultPoint(centerx - ratio * dx, centery - ratio * dy);

	dx = cornerPoints[1].x() - cornerPoints[3].x();
	dy = cornerPoints[1].y() - cornerPoints[3].y();
	centerx = (cornerPoints[1].x() + cornerPoints[3].x()) / 2.0f;
	centery = (cornerPoints[1].y() + cornerPoints[3].y()) / 2.0f;
	cornerPoints[1] = ResultPoint(centerx + ratio * dx, centery + ratio * dy);
	cornerPoints[3] = ResultPoint(centerx - ratio * dx, centery - ratio * dy);
}


/**
* Finds the corners of a bull-eye centered on the passed point.
* This returns the centers of the diagonal points just outside the bull's eye
* Returns [topRight, bottomRight, bottomLeft, topLeft]
*
* @param pCenter Center point
* @return The corners of the bull-eye
* @throws NotFoundException If no valid bull-eye can be found
*/
static bool GetBullsEyeCorners(const BitMatrix& image, const PixelPoint& pCenter, std::array<ResultPoint, 4>& result, bool& compact, int& nbCenterLayers)
{
	PixelPoint pina = pCenter;
	PixelPoint pinb = pCenter;
	PixelPoint pinc = pCenter;
	PixelPoint pind = pCenter;

	bool color = true;
	for (nbCenterLayers = 1; nbCenterLayers < 9; nbCenterLayers++) {
		PixelPoint pouta = GetFirstDifferent(image, pina, color, 1, -1);
		PixelPoint poutb = GetFirstDifferent(image, pinb, color, 1, 1);
		PixelPoint poutc = GetFirstDifferent(image, pinc, color, -1, 1);
		PixelPoint poutd = GetFirstDifferent(image, pind, color, -1, -1);

		//d      a
		//
		//c      b

		if (nbCenterLayers > 2) {
			float q = Distance(poutd, pouta) * nbCenterLayers / (Distance(pind, pina) * (nbCenterLayers + 2));
			bool bWhiteOrBlack = IsWhiteOrBlackRectangle(image, pouta, poutb, poutc, poutd);
			//araj
			/*if (q < 0.75 || q > 1.25 || !bWhiteOrBlack) {
				break;
			}*/
			if (q < 0.7 || q > 1.3 || !bWhiteOrBlack) {
				break;
			}
		}

		pina = pouta;
		pinb = poutb;
		pinc = poutc;
		pind = poutd;

		color = !color;
	}

	if (nbCenterLayers != 5 && nbCenterLayers != 7) {
		return false;
	}

	compact = nbCenterLayers == 5;

	// Expand the square by .5 pixel in each direction so that we're on the border
	// between the white square and the black square
	result[0] = ResultPoint(pina.x + 0.5f, pina.y - 0.5f);
	result[1] = ResultPoint(pinb.x + 0.5f, pinb.y + 0.5f);
	result[2] = ResultPoint(pinc.x - 0.5f, pinc.y + 0.5f);
	result[3] = ResultPoint(pind.x - 0.5f, pind.y - 0.5f);

	// Expand the square so that its corners are the centers of the points
	// just outside the bull's eye.
	ExpandSquare(result, static_cast<float>(2 * nbCenterLayers - 3), static_cast<float>(2 * nbCenterLayers));
	return true;
}

/**
* Finds a candidate center point of an Aztec code from an image
*
* @return the center point
*/
static PixelPoint GetMatrixCenter(const BitMatrix& image)
{
	//Get a white rectangle that can be the border of the matrix in center bull's eye or
	ResultPoint pointA, pointB, pointC, pointD;
	if (!WhiteRectDetector::Detect(image, pointA, pointB, pointC, pointD)) {
		// This exception can be in case the initial rectangle is white
		// In that case, surely in the bull's eye, we try to expand the rectangle.
		int cx = image.width() / 2;
		int cy = image.height() / 2;
		pointA = GetFirstDifferent(image, { cx + 7, cy - 7 }, false, 1, -1).toResultPoint();
		pointB = GetFirstDifferent(image, { cx + 7, cy + 7 }, false, 1, 1).toResultPoint();
		pointC = GetFirstDifferent(image, { cx - 7, cy + 7 }, false, -1, 1).toResultPoint();
		pointD = GetFirstDifferent(image, { cx - 7, cy - 7 }, false, -1, -1).toResultPoint();
	}

	//Compute the center of the rectangle
	int cx = RoundToNearest((pointA.x() + pointD.x() + pointB.x() + pointC.x()) / 4.0f);
	int cy = RoundToNearest((pointA.y() + pointD.y() + pointB.y() + pointC.y()) / 4.0f);

	// Redetermine the white rectangle starting from previously computed center.
	// This will ensure that we end up with a white rectangle in center bull's eye
	// in order to compute a more accurate center.
	if (!WhiteRectDetector::Detect(image, 15, cx, cy, pointA, pointB, pointC, pointD)) {
		// This exception can be in case the initial rectangle is white
		// In that case we try to expand the rectangle.
		pointA = GetFirstDifferent(image, { cx + 7, cy - 7 }, false, 1, -1).toResultPoint();
		pointB = GetFirstDifferent(image, { cx + 7, cy + 7 }, false, 1, 1).toResultPoint();
		pointC = GetFirstDifferent(image, { cx - 7, cy + 7 }, false, -1, 1).toResultPoint();
		pointD = GetFirstDifferent(image, { cx - 7, cy - 7 }, false, -1, -1).toResultPoint();
	}

	// Recompute the center of the rectangle
	cx = RoundToNearest((pointA.x() + pointD.x() + pointB.x() + pointC.x()) / 4.0f);
	cy = RoundToNearest((pointA.y() + pointD.y() + pointB.y() + pointC.y()) / 4.0f);

	return{ cx, cy };
}

static int GetDimension(bool compact, int nbLayers)
{
	if (compact) {
		return 4 * nbLayers + 11;
	}
	if (nbLayers <= 4) {
		return 4 * nbLayers + 15;
	}
	return 4 * nbLayers + 2 * ((nbLayers - 4) / 8 + 1) + 15;
}


/**
* Gets the Aztec code corners from the bull's eye corners and the parameters.
*
* @param bullsEyeCorners the array of bull's eye corners
* @return the array of aztec code corners
*/
static void GetMatrixCornerPoints(std::array<ResultPoint, 4>& bullsEyeCorners, bool compact, int nbLayers, int nbCenterLayers)
{
	ExpandSquare(bullsEyeCorners, static_cast<float>(2 * nbCenterLayers), static_cast<float>(GetDimension(compact, nbLayers)));
}

//araj

//get relative depth data
static void get_relative_depth(int* depth, int dim, float step) {
	int sind = -1;
	int min_depth = 999;
	int i, j;
	int cd, nd;
	int istep = step + 0.5;

	// get shallowest index
	for (i = 0; i < dim; i++) {
		if (min_depth > depth[i]) {
			min_depth = depth[i];
			sind = i;
		}
	}

	// scan left from sind
	for (i = sind; i > 0; i--) {
		j = i - 1;
		cd = depth[i];
		nd = depth[j];
		int  nd2 = nd;
		
		int t = nd / step;
		t = nd - step * t + 0.5;
		t = t % istep;

		int t2 = cd % istep;

		if (t > t2) {
			if (t - t2 < step / 2)
				nd = cd - t2 + t;
			else
				nd = cd - t2 + t - 6;
		}
		else {
			if (t2 - t < step / 2)
				nd = cd - t2 + t;
			else
				nd = cd - t2 + t + 6;
		}
		if (abs(nd - cd) > 1)
			nd = cd;
		if (nd < 0) nd = 0;

		//depth[j] = nd;
		depth[j] = MIN(nd, nd2);
	}
	
	// scan right from sind
	for (i = sind; i < dim; i++) {
		j = i + 1;
		cd = depth[i];
		nd = depth[j];
		int nd2 = nd;

		int t = nd / step;
		t = nd - step * t + 0.5;
		t = t % istep;

		int t2 = cd % istep;

		if (t > t2) {
			if (t - t2 < step / 2)
				nd = cd - t2 + t;
			else
				nd = cd - t2 + t - 6;
		}
		else {
			if (t2 - t < step / 2)
				nd = cd - t2 + t;
			else
				nd = cd - t2 + t + 6;
		}
		if (abs(nd - cd) > 1)
			nd = cd;
		if (nd < 0) nd = 0;

		depth[j] = MIN(nd, nd2);
		//depth[j] = nd;
	}
}

// get start differences from left, right, top, bottom edges
static void get_differences(const BitMatrix& img, int* left, int* right, int* top, int* bottom, int dim) {
	float w = img.width();
	float step = w / (float)dim;

	// get raw depth data
	for (int i = 0; i < dim; i++) {
		int s, e;
		int depth;

		depth = MIN(w, step * 20);
		s = step * i + 0.5;
		e = step * (i + 1) + 0.5;

		// get left 
		for (int d = 0; d < depth; d++) {
			int sum = 0;
			for (int j = s; j < e; j++) {
				sum += img.get(d, j);
			}
			if (sum > (e - s) / 2) {
				left[i] = d;
				break;
			}
		}

		// get right
		for (int d = 0; d < depth; d++) {
			int sum = 0;
			for (int j = s; j < e; j++) {
				sum += img.get(w - 1 - d, j);
			}
			if (sum > (e - s) / 2) {
				right[i] = d;
				break;
			}
		}

		//get top
		for (int d = 0; d < depth; d++) {
			int sum = 0;
			for (int j = s; j < e; j++) {
				sum += img.get(j, d);
			}
			if (sum > (e - s) / 2) {
				top[i] = d;
				break;
			}
		}

		// get bottom
		for (int d = 0; d < depth; d++) {
			int sum = 0;
			for (int j = s; j < e; j++) {
				sum += img.get(j, w - 1 - d);
			}
			if (sum > (e - s) / 2) {
				bottom[i] = d;
				break;
			}
		}
	}

	get_relative_depth(left, dim, step);
	get_relative_depth(right, dim, step);
	get_relative_depth(top, dim, step);
	get_relative_depth(bottom, dim, step);
}

// my abs functin
static int iabs(int x) {
	if (x > 0) return x;
	return (-x);
}
// get border position
// direction (0:Right, 1:Bottom, 2:Left, 3:Top)
static float get_border_pos(const BitMatrix& img, int direction, float cx, float cy, float delta, int w, int h, float&tscore) {
	float res, res2;
	bool cval;
	float thres1 = 0.3, thres2 = 0.7;
	float delta2;

	tscore = 0;

	// Right
	if (direction == 0) {
		res = cx + delta/2;
		res = MIN(w - 1.5, MAX(0, res));
		cy = MIN(h - 1.5, MAX(0, cy));
		res2 = res;
		cval = img.get(res + 0.5f, cy+0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1;
				break;
			}
			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res--;
				break;
			}
			res++;
		}

		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1;
				break;
			}
			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;
		}

		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta;
			int n = fn + 0.5;
			tscore = fn - n;
			if (tscore < 0) tscore = 1 - tscore;

			if (n < 2) {

			}
			else {
				delta2 = (res - res2+1) / n;
				n = (cx - res2+1) / delta2 + 0.5 + 1;
				res = MIN(w - 1, res2 + delta2 * n);
			}
		}
	}

	// Left
	else if (direction == 2) {
		res = cx - delta / 2;
		res = MIN(w - 1.5, MAX(0, res));
		cy = MIN(h - 1.5, MAX(0, cy));
		res2 = res;
		cval = img.get(res + 0.5f, cy + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1;
				break;
			}

			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1;
				break;
			}

			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;
		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta;
			int n = fn + 0.5;
			tscore = fn - n;
			if (tscore < 0) tscore = 1 - tscore;
			if (n < 2) {

			}
			else {
				delta2 = (res2 - res+1) / n;
				n = (cx - res+1) / delta2 + 0.5 - 1;
				res = MIN(w - 1, res + delta2 * n);
			}
		}
	}

	// Top
	else if (direction == 3) {
		res = cy - delta / 2;
		res = MIN(h - 1.5, MAX(0, res));
		cx = MIN(w - 1.5, MAX(0, cx));
		res2 = res;
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h - 0.5f) {
				res = h - 1;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
			
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;

		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta;
			int n = fn + 0.5;
			tscore = fn - n;
			if (tscore < 0) tscore = 1 - tscore;
			if (n < 2) {
				
			}
			else {
				delta2 = (res2 - res+1) / n;
				n = (cy - res+1) / delta2 + 0.5 - 1;
				res = MIN(h - 1, res + delta2 * n);
			}
		}
	}
	// Bottom
	else if (direction == 1) {
		res = cy + delta / 2 ;
		res = MIN(h - 1.5, MAX(0, res));
		cx = MIN(w - 1.5, MAX(0, cx));
		res2 = res;
		
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h-0.5f) {
				res = h - 1;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res--;
				break;
			}
			res++;
			
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;

		}
		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta;
			int n = fn + 0.5;
			tscore = fn - n;
			if (tscore < 0) tscore = 1 - tscore;
			if (n < 2) {

			}
			else {
				delta2 = (res - res2+1) / n;
				n = (cy - res2+1) / delta2 + 0.5 + 1;
				res = MIN(w - 1, res2 + delta2 * n);
			}
		}
	}

	if (direction == 0 && res - cx >= delta + (delta - 1) / 2) {
		res = MIN(w - 1.5, cx + delta);
	}
	else if (direction == 2 && cx - res >= delta + (delta - 1) / 2) {
		res = MAX(0, cx - delta);
	}
	else if (direction == 1 && res - cy >= delta + (delta - 1) / 2) {
		res = MIN(h-1.5, cy + delta);
	}
	else if (direction == 3 && cy - res >= delta + (delta - 1) / 2) {
		res = MAX(0, cy - delta);
	}
	
	return res;
}

static float get_border_pos3(const BitMatrix& img, int direction, float cx, float cy, float delta, int w, int h, float& tscore) {
	float res, res2;
	bool cval;
	float thres1 = 0.3, thres2 = 0.7;
	float delta2;
	float fadd = 0.75;

	tscore = 0;
	// Right
	if (direction == 0) {
		res2 = res = MIN(w-1.5, MAX(0, cx));
		cy = MIN(h - 1.5, MAX(0, cy));
		cval = img.get(res + 0.5f, cy + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1.f;
				break;
			}
			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res--;
				break;
			}
			res++;
		}

		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1.f;
				break;
			}
			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;
		}

		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta;
			int n = fn + 0.5;
			if (fn > 8)
				n = fn + fadd;
			tscore = fn - n;
			if (tscore < 0) tscore = -tscore;

			if (n == 0) {
				res = cx;
			}
			else if (n == 1) {
				res = (res + res2) / 2.f;
			}
			else {
				delta2 = (res - res2 + 1.f) / n;
				n = (cx - res2) / delta2;
				res = MIN(w - 1, res2 + delta2 * n + delta2 / 2.f);
			}
		}
	}

	// Left
	else if (direction == 2) {
		res2 = res = MIN(w - 1.5, MAX(0, cx));
		cy = MIN(h - 1.5, MAX(0, cy));
		cval = img.get(res + 0.5f, cy + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1.f;
				break;
			}

			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1.f;
				break;
			}

			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;
		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta;
			int n = fn + 0.5;
			if (fn > 8)
				n = fn + fadd;
			tscore = fn - n;
			if (tscore < 0) tscore = -tscore;

			if (n == 0) {
				res = cx;
			}
			else if (n == 1) {
				res = (res + res2) / 2.f;
			}
			else {
				delta2 = (res2 - res + 1.f) / n;
				n = (cx - res) / delta2;
				res = MIN(w - 1, res + delta2 * n + delta2 / 2.f);
			}
		}
	}

	// Top
	else if (direction == 3) {
		res2 = res = MIN(h - 1.5, MAX(0, cy));
		cx = MIN(w - 1.5, MAX(0, cx));
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h - 0.5f) {
				res = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;

		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta;
			int n = fn + 0.5;
			if (fn > 8)
				n = fn + fadd;
			tscore = fn - n;
			if (tscore < 0) tscore = -tscore;

			if (n == 0) {
				res = cy;
			}
			else if (n == 1) {
				res = (res + res2) / 2.f;
			}
			else {
				delta2 = (res2 - res + 1.f) / n;
				n = (cy - res) / delta2;
				res = MIN(h - 1, res + delta2 * n + delta2 / 2.f);
			}
		}
	}
	// Bottom
	else if (direction == 1) {
		res2 = res = MIN(h - 1.5, MAX(0, cy));
		cx = MIN(w - 1.5, MAX(0, cx));
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h - 0.5f) {
				res = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res--;
				break;
			}
			res++;

		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;

		}
		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta;
			int n = fn + 0.5;
			if (fn > 8)
				n = fn + fadd;
			tscore = fn - n;
			if (tscore < 0) tscore = -tscore;
			if (n == 0) {
				res = cy;
			}
			else if (n == 1) {
				res = (res + res2) / 2.f;
			}
			else {
				delta2 = (res - res2 + 1.f) / n;
				n = (cy - res2) / delta2;
				res = MIN(w - 1, res2 + delta2 * n + delta2 / 2.f);
			}
		}
	}
	return res;
}

static float get_border_pos2(const BitMatrix& img, int direction, float cx, float cy, float delta, int w, int h, float& tscore) {
	float res, res2;
	bool cval;
	float thres1 = 0.3, thres2 = 0.7;
	float delta2 = delta / 2;
	float fadd_8 = 0.5;

	tscore = 0;
	// Right
	if (direction == 0) {
		res2 = res = MIN(w - 1.5, MAX(0, cx));
		cy = MIN(h - 1.5, MAX(0, cy));
		cval = img.get(res + 0.5f, cy + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1.f;
				break;
			}
			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res--;
				break;
			}
			res++;
		}

		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1.f;
				break;
			}
			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;
		}

		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta2;
			if (fn > 16)
				fn += fadd_8;
			int n = fn + 0.5;
			if (n % 2 == 1) {
				if (fn - n + 1 > n + 1 - fn)
					n++;
				else
					n--;
			}

			if (n == 0) {
				res = cx;
			}
			else if (n == 2) {
				res = (res + res2) / 2.f;
			}
			else {
				fn = (res - res2 + 1.f) / n;
				float fn2 = fn;
				fn = (cx - res2) / fn;
				n = (int)(fn + 0.5);
				if (n == 0) {
					n = 1;
				}
				else if (n % 2 == 0) {
					if (fn - n + 1 > n + 1 - fn)
						n++;
					else
						n--;
				}

				res = MIN(w - 1, res2 + fn2 * n);
			}
		}
	}

	// Left
	else if (direction == 2) {
		res2 = res = MIN(w - 1.5, MAX(0, cx));
		cy = MIN(h - 1.5, MAX(0, cy));
		cval = img.get(res + 0.5f, cy + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= w - 0.5f) {
				res = w - 1.f;
				break;
			}

			if (img.get(res + 0.5f, cy + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= w - 0.5f) {
				res2 = w - 1.f;
				break;
			}

			if (img.get(res2 + 0.5f, cy + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;
		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta2;
			if (fn > 16)
				fn += fadd_8;

			int n = fn + 0.5;
			if (n % 2 == 1) {
				if (fn - n + 1 > n + 1 - fn)
					n++;
				else
					n--;
			}

			if (n == 0) {
				res = cx;
			}
			else if (n == 2) {
				res = (res + res2) / 2.f;
			}
			else {
				fn = (res2 - res + 1.f) / n;
				float fn2 = fn;
				fn = (cx - res) / fn;
				n = (int)(fn + 0.5);
				if (n == 0) {
					n = 1;
				}
				else if (n % 2 == 0) {
					if (fn - n + 1 > n + 1 - fn)
						n++;
					else
						n--;
				}

				res = MIN(w - 1, res + fn2 * n);
			}
		}
	}

	// Top
	else if (direction == 3) {
		res2 = res = MIN(h - 1.5, MAX(0, cy));
		cx = MIN(w - 1.5, MAX(0, cx));
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h - 0.5f) {
				res = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res++;
				break;
			}
			res--;
		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2--;
				break;
			}
			res2++;

		}
		if (res2 <= res) {
			res = res2;
		}
		else {
			float fn = (res2 - res + 1.f) / delta2;
			if (fn > 16)
				fn += fadd_8;

			int n = fn + 0.5;
			if (n % 2 == 1) {
				if (fn - n + 1 > n + 1 - fn)
					n++;
				else
					n--;
			}

			if (n == 0) {
				res = cy;
			}
			else if (n == 2) {
				res = (res + res2) / 2.f;
			}
			else {
				fn = (res2 - res + 1.f) / n;
				float fn2 = fn;
				fn = (cy - res) / fn;
				n = (int)(fn + 0.5);
				if (n == 0) {
					n = 1;
				}
				else if (n%2 == 0){
					if (fn - n + 1 > n + 1 - fn)
						n++;
					else
						n--;
				}

				res = MIN(w - 1, res + fn2 * n);
			}
		}
	}
	// Bottom
	else if (direction == 1) {
		res2 = res = MIN(h - 1.5, MAX(0, cy));
		cx = MIN(w - 1.5, MAX(0, cx));
		cval = img.get(cx + 0.5f, res + 0.5f);
		for (;;) {
			if (res < 0) {
				res = 0;
				break;
			}
			else if (res >= h - 0.5f) {
				res = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res + 0.5f) != cval) {
				res--;
				break;
			}
			res++;

		}
		for (;;) {
			if (res2 < 0) {
				res2 = 0;
				break;
			}
			else if (res2 >= h - 0.5f) {
				res2 = h - 1.f;
				break;
			}

			if (img.get(cx + 0.5f, res2 + 0.5f) != cval) {
				res2++;
				break;
			}
			res2--;

		}
		if (res2 >= res) {
			res = res2;
		}
		else {
			float fn = (res - res2 + 1.f) / delta2, fn2;
			if (fn > 16)
				fn += fadd_8;

			int n = fn + 0.5;
			if (n % 2 == 1) {
				if (fn - n + 1 > n + 1 - fn)
					n++;
				else
					n--;
			}

			if (n == 0) {
				res = cy;
			}
			else if (n == 2) {
				res = (res + res2) / 2.f;
			}
			else {
				fn = (res - res2 + 1.f) / n;
				fn2 = fn;
				fn = (cy - res2) / fn;
				n = (int)(fn + 0.5);
				if (n == 0) {
					n = 1;
				}
				else if (n % 2 == 0) {
					if (fn - n + 1 > n + 1 - fn)
						n++;
					else
						n--;
				}

				res = MIN(w - 1, res2 + fn2 * n);
			}
		}
	}

	if (direction == 0 || direction == 2) {
		if (res - cx >= (delta-0.5) / 2 || cx - res >= (delta - 0.5) / 2)
			res = MIN(w - 1.5, MAX(0, cx));
	}
	else {
		if (res - cy >= (delta - 0.5) / 2 || cy - res >= (delta - 0.5) / 2)
			res = MIN(h - 1.5, MAX(0, cy));
	}
	return res;
}
// get bit data from bitmatrix
static BitMatrix get_bits(const BitMatrix& img, int cx, int cy, int dimension, int w, int h, Mat mimg) {
	float thres = 999;
	BitMatrix res(dimension, dimension);
	Point CL[200], CR[200], CT[200], CB[200];
	Point PL[200], PR[200], PT[200], PB[200];
	int offset = dimension / 2;
	float delta = (float)w / (float)dimension;
	float tscore;

	//starting prev points
	PL[offset] = Point(cx - delta/2, cy);
	PR[offset] = Point(cx + delta / 2, cy);
	PT[offset] = Point(cx, cy - delta / 2);
	PB[offset] = Point(cx, cy + delta / 2);
	res.set(offset, offset);

	Mat m2;
	mimg.copyTo(m2);
	uchar* data = m2.data;

	//loop from 1 ~ offset(41:large)
	for (int i = 1; i <= offset; i++) {
		//get points from offset-i ~ offset + i
		offset = offset;
		// get Right
		for (int j = offset - i; j <= offset + i; j++) {
			//get CL, CR, CT, CB from PL, PR, PT, TB
			
				//get U, D
			float u, d, tm, tx, ntx;
			if (j == offset - i) {
				//d = PT[offset + i - 1].y;
				d = PT[offset + i - 1].y - delta/2;

				tm = PT[offset + i - 1].x + delta;
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				tm = get_border_pos2(img, 1, tm, d, delta, w, h, tscore);
				if (tscore > thres)
					tm = d;
				

				tx = PR[offset - i + 1].x;
			}
			else if (j == offset + i) {
				//u = PB[offset + i - 1].y;
				u = PB[offset + i - 1].y + delta/2;

				tm = PB[offset + i - 1].x + delta;
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				tm = get_border_pos2(img, 1, tm, u, delta, w, h, tscore);
				if (tscore > thres)
					tm = u;
				tx = PR[offset + i - 1].x;
			}
			else {
				d = PR[j].y + delta / 2;
				tm = PR[j].x + delta/2;
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				float tm1 = get_border_pos2(img, 3, tm, PR[j].y, delta, w, h, tscore);
				if (i < offset - 1) {
					float tm2 = get_border_pos2(img, 3, tm + delta, PR[j].y, delta, w, h, tscore);
					if (abs(PR[j].y - tm1) > abs(PR[j].y - tm2))
						tm = tm2;
					else
						tm = tm1;
				}
				else
					tm = tm1;
				tx = PR[j].x;
			}
			//tm = (u + d) / 2.f;
			ntx = get_border_pos(img, 0, tx, tm, delta, w, h, tscore);
			if (tscore > thres && 0)
				ntx = tx + delta;

			CR[j] = Point(round(ntx), round(tm));

			int intx, itm;
			intx = round((tx + ntx) / 2.f);
			itm = round(tm);

			int sum = 0;
			int sx, ex, sy, ey;
			sx = MIN(w - 1, MAX(0, intx - 1));
			ex = MIN(w - 1, MAX(0, intx + 2));
			sy = MIN(h - 1, MAX(0, itm - 1));
			ey = MIN(h - 1, MAX(0, itm + 2));
			for (int ix = sx; ix < ex; ix++)
				for (int iy = sy; iy < ey; iy++)
					sum += img.get(ix, iy);
			if (sum >= (ex-sx) * (ey - sy)/2)
				res.set(offset + i, j);
		}

		// get Left
		for (int j = offset - i; j <= offset + i; j++) {
			//get CL, CR, CT, CB from PL, PR, PT, TB

				//get U, D
			float u, d, tm, tx, ntx;
			if (j == offset - i) {
				d = PT[offset - i + 1].y;
				tm = PT[offset - i + 1].x - delta;
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				tm = get_border_pos2(img, 3, tm, PT[offset - i + 1].y - delta/2, delta, w, h, tscore);
				if (tscore > thres)
					tm = PT[offset - i + 1].y - delta / 2;

				tx = PL[offset - i + 1].x;
			}
			else if (j == offset + i) {
				u = PB[offset - i + 1].y;
				tm = PB[offset - i + 1].x - delta;
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				tm = get_border_pos2(img, 1, tm, PB[offset - i + 1].y + delta/2, delta, w, h, tscore);
				if (tscore > thres)
					tm = PB[offset - i + 1].y + delta / 2;

				tx = PL[offset + i - 1].x;
			}
			else {
				d = PL[j].y + delta / 2 ;
				tm = PL[j].x - delta / 2;
				//u = get_border_pos(img, 3, tm, d, delta, w, h);
				//d = get_border_pos(img, 1, tm, u, delta, w, h);
				float tm1 = get_border_pos2(img, 3, tm, PL[j].y, delta, w, h, tscore);
				if (i < offset - 1) {
					float tm2 = get_border_pos2(img, 3, tm - delta, PL[j].y, delta, w, h, tscore);
					if (abs(PL[j].y - tm1) > abs(PL[j].y - tm2))
						tm = tm2;
					else
						tm = tm1;
				}
				else
					tm = tm1;

				tx = PL[j].x;
			}
			//tm = (u + d) / 2.f;
			ntx = get_border_pos(img, 2, tx, tm, delta, w, h, tscore);
			if (tscore > thres && 0)
				ntx = tx - delta;

			CL[j] = Point(round(ntx), round(tm));

			if ((int)(ntx + 0.5) == 193 && int(tm + 0.5) == 277) {
				offset = offset;
			}

			int intx, itm;
			intx = round((tx + ntx) / 2.f);
			itm = round(tm);

			int sum = 0;
			int sx, ex, sy, ey;
			sx = MIN(w - 1, MAX(0, intx - 1));
			ex = MIN(w - 1, MAX(0, intx + 2));
			sy = MIN(h - 1, MAX(0, itm - 1));
			ey = MIN(h - 1, MAX(0, itm + 2));
			for (int ix = sx; ix < ex; ix++)
				for (int iy = sy; iy < ey; iy++)
					sum += img.get(ix, iy);
			if (sum >= (ex - sx) * (ey - sy) / 2)
				res.set(offset - i, j);

			
		}

		// get Bottom
		for (int j = offset - i; j <= offset + i; j++) {
			//get CL, CR, CT, CB from PL, PR, PT, TB

				//get l, r
			float l, r, tm, ty, nty;
			if (j == offset - i) {
				r = PL[offset + i - 1].x;
				tm = PL[offset + i - 1].y + delta;
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				tm = get_border_pos2(img, 2, PL[offset + i - 1].x - delta/2, tm, delta, w, h, tscore);
				if (tscore > thres)
					tm = PL[offset + i - 1].x - delta / 2;

				ty = PB[offset - i + 1].y;
			}
			else if (j == offset + i) {
				l = PR[offset + i - 1].x;
				tm = PR[offset + i - 1].y + delta;
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				tm = get_border_pos2(img, 0, PR[offset + i - 1].x + delta/2, tm, delta, w, h, tscore);
				if (tscore > thres)
					tm = PR[offset + i - 1].x;

				ty = PB[offset + i - 1].y;
			}
			else {
				r = PB[j].x + delta / 2;
				tm = PB[j].y + delta / 2;
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				float tm1 = get_border_pos2(img, 2, PB[j].x, tm, delta, w, h, tscore);
				if (i < offset - 1) {
					float tm2 = get_border_pos2(img, 2, PB[j].x, tm + delta, delta, w, h, tscore);
					if (abs(PB[j].x - tm1) > abs(PB[j].x - tm2))
						tm = tm2;
					else
						tm = tm1;
				}
				else
					tm = tm1;

				ty = PB[j].y;
			}
			//tm = (l + r) / 2.f;
			nty = get_border_pos(img, 1, tm, ty, delta, w, h, tscore);
			if (tscore > thres && 0)
				nty = ty + delta;

			CB[j] = Point(round(tm), round(nty));

			int inty, itm;
			
			inty = round((ty + nty) / 2.f);
			itm = round(tm);

			int sum = 0;
			int sx, ex, sy, ey;
			sx = MIN(w - 1, MAX(0, itm - 1));
			ex = MIN(w - 1, MAX(0, itm + 2));
			sy = MIN(h - 1, MAX(0, inty - 1));
			ey = MIN(h - 1, MAX(0, inty + 2));
			for (int ix = sx; ix < ex; ix++)
				for (int iy = sy; iy < ey; iy++)
					sum += img.get(ix, iy);
			if (sum >= (ex - sx) * (ey - sy) / 2)
				res.set(j, offset + i);

		}

		// get Top
		for (int j = offset - i; j <= offset + i; j++) {
			//get CL, CR, CT, CB from PL, PR, PT, TB

				//get l, r
			float l, r, tm, ty, nty;
			if (j == offset - i) {
				r = PL[offset - i + 1].x;
				tm = PL[offset - i + 1].y - delta;
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				tm = get_border_pos2(img, 2, PL[offset - i + 1].x - delta/2, tm, delta, w, h, tscore);
				if (tscore > thres)
					tm = PL[offset - i + 1].x - delta / 2;

				ty = PT[offset - i + 1].y;
			}
			else if (j == offset + i) {
				l = PR[offset - i + 1].x;
				tm = PR[offset - i + 1].y - delta;
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				tm = get_border_pos2(img, 0, PR[offset - i + 1].x + delta/2, tm, delta, w, h, tscore);
				if (tscore > thres)
					tm = PR[offset - i + 1].x + delta / 2;

				ty = PT[offset + i - 1].y;
			}
			else {
				r = PT[j].x + delta / 2;
				tm = PT[j].y - delta / 2;
				//l = get_border_pos(img, 2, r, tm, delta, w, h);
				//r = get_border_pos(img, 0, l, tm, delta, w, h);
				float tm1 = get_border_pos2(img, 2, PT[j].x, tm, delta, w, h, tscore);
				if (i < offset - 1) {
					float tm2 = get_border_pos2(img, 2, PT[j].x, tm - delta, delta, w, h, tscore);
					if (abs(PT[j].x - tm1) > abs(PT[j].x - tm2))
						tm = tm2;
					else
						tm = tm1;
				}
				else
					tm = tm1;

				ty = PT[j].y;
			}
			//tm = (l + r) / 2.f;
			nty = get_border_pos(img, 3, tm, ty, delta, w, h, tscore);
			if (tscore > thres && 0)
				nty = ty - delta;

			CT[j] = Point(round(tm), round(nty));

			if ((int)(tm + 0.5) == 125 && (int)(nty + 0.5) == 131) {
				offset = offset;
			}

			int inty, itm;
			inty = round((ty + nty) / 2.f);
			itm = round(tm);
			
			int sum = 0;
			int sx, ex, sy, ey;
			sx = MIN(w - 1, MAX(0, itm - 1));
			ex = MIN(w - 1, MAX(0, itm + 2));
			sy = MIN(h - 1, MAX(0, inty - 1));
			ey = MIN(h - 1, MAX(0, inty + 2));
			for (int ix = sx; ix < ex; ix++)
				for (int iy = sy; iy < ey; iy++)
					sum += img.get(ix, iy);
			if (sum >= (ex - sx) * (ey - sy) / 2)
				res.set(j, offset - i);
		}

		//update midpoints
		if (true)
		{
			CL[offset + i].y = (PB[offset - i + 1].y + CB[offset - i].y + 1) / 2;
			CL[offset - i].y = (PT[offset - i + 1].y + CT[offset - i].y + 1) / 2;

			CR[offset + i].y = (PB[offset + i - 1].y + CB[offset + i].y + 1) / 2;
			CR[offset - i].y = (PT[offset + i - 1].y + CT[offset + i].y + 1) / 2;

			CT[offset + i].x = (PR[offset - i + 1].x + CR[offset - i].x + 1) / 2;
			CT[offset - i].x = (PL[offset - i + 1].x + CL[offset - i].x + 1) / 2;

			CB[offset + i].x = (PR[offset + i - 1].x + CR[offset + i].x + 1) / 2;
			CB[offset - i].x = (PL[offset + i - 1].x + CL[offset + i].x + 1) / 2;

			float dthres = delta / 2;
			float dthres2 = dthres + delta;
			float v;
			for (int j = offset - i + 1; j <= offset + i - 1; j++) {
				
				// Left
				v = abs(CL[j].x - CL[j - 1].x);
				if (v > dthres) {
					if (abs(CL[j].x - CL[j + 1].x) > abs(CL[j - 1].x - CL[j + 1].x)) {
						CL[j].x = CL[j + 1].x;
					}
					else {
						CL[j - 1].x = CL[j].x;
					}
				}
				v = abs(CL[j].y - CL[j - 1].y);
				if (v < dthres || v > dthres2) {
					v = abs(CL[j + 1].y - CL[j].y);
					if (v < dthres || v > dthres2) {
						CL[j].y = MIN(h-1, CL[j].y + delta);
					}
					else {
						CL[j - 1].y = MAX(0, CL[j].y - delta);
					}
				}
				

				// Right
				v = abs(CR[j].x - CR[j - 1].x);
				if (v > dthres) {
					if (abs(CR[j].x - CR[j + 1].x) > abs(CR[j - 1].x - CR[j + 1].x)) {
						CR[j].x = CR[j + 1].x;
					}
					else {
						CR[j - 1].x = CR[j].x;
					}
				}
				v = abs(CR[j].y - CR[j - 1].y);
				if (v < dthres || v > dthres2) {
					v = abs(CR[j + 1].y - CR[j].y);
					if (v < dthres || v > dthres2) {
						CR[j].y = MIN(h-1, CR[j-1].y + delta);
					}
					else {
						CR[j - 1].y = MAX(0, CR[j].y - delta);
					}
				}

				// Top
				v = abs(CT[j].y - CT[j - 1].y);
				if (v > dthres) {
					if (abs(CT[j].y - CT[j + 1].y) > abs(CT[j - 1].y - CT[j + 1].y)) {
						CT[j].y = CT[j + 1].y;
					}
					else {
						CT[j - 1].y = CT[j].y;
					}
				}
				v = abs(CT[j].x - CT[j - 1].x);
				if (v < dthres || v > dthres2) {
					v = abs(CT[j + 1].x - CT[j].x);
					if (v < dthres || v > dthres2) {
						CT[j].x = MIN(w-1, CT[j-1].x + delta);
					}
					else {
						CT[j - 1].x = MAX(0, CT[j].x - delta);
					}
				}

				// Bottom
				v = abs(CB[j].y - CB[j - 1].y);
				if (v > dthres) {
					if (abs(CB[j].y - CB[j + 1].y) > abs(CB[j - 1].y - CB[j + 1].y)) {
						CB[j].y = CB[j + 1].y;
					}
					else {
						CB[j - 1].y = CB[j].y;
					}
				}
				v = abs(CB[j].x - CB[j - 1].x);
				if (v < dthres || v > dthres2) {
					v = abs(CB[j + 1].x - CB[j].x);
					if (v < dthres || v > dthres2) {
						CB[j].x = MIN(w-1, CB[j-1].x + delta);
					}
					else {
						CB[j - 1].x = MAX(0, CB[j].x - delta);
					}
				}

				if (CL[j].x == 91 && CL[j + 1].y == 229) {
					offset = offset;
				}
			}

			// Left
			if (abs(CL[offset + i].x - CL[offset + i - 1].x) > dthres) {
				CL[offset + i].x = CL[offset + i - 1].x;
			}
			v = abs(CL[offset + i].y - CL[offset + i - 1].y);
			if (v < dthres || v > dthres2) {
				CL[offset + i].y = MIN(h - 1, CL[offset + i - 1].y + delta);
			}
			//Right
			if (abs(CR[offset + i].x - CR[offset + i - 1].x) > dthres) {
				CR[offset + i].x = CR[offset + i - 1].x;
			}
			v = abs(CR[offset + i].y - CR[offset + i - 1].y);
			if (v < dthres || v > dthres2) {
				CR[offset + i].y = MIN(h-1, CR[offset + i - 1].y + delta);
			}
			//Top
			if (abs(CT[offset + i].y - CT[offset + i - 1].y) > dthres) {
				CT[offset + i].y = CT[offset + i - 1].y;
			}
			v = abs(CT[offset + i].x - CT[offset + i - 1].x);
			if (v < dthres || v > dthres2) {
				CT[offset + i].x = MIN(w-1, CT[offset + i - 1].x + delta);
			}
			//Bottom
			if (abs(CB[offset + i].y - CB[offset + i - 1].y) > dthres) {
				CB[offset + i].y = CB[offset + i - 1].y;
			}
			v = abs(CB[offset + i].x - CB[offset + i - 1].x);
			if (v < dthres || v > dthres2) {
				CB[offset + i].x = MIN(w-1, CB[offset + i - 1].x + delta);
			}
		}
		for (int j = 0; j < 200; j++) {
			PL[j] = CL[j];
			PR[j] = CR[j];
			PB[j] = CB[j];
			PT[j] = CT[j];
		}

		
		
		for (int j = offset - i; j <= offset + i; j++) {
			int cur = 3*(PL[j].x + PL[j].y * w);
			data[cur] = 0x00;
			data[cur + 1] = 0xff;
			data[cur + 2] = 0x00;

			cur = 3*(PR[j].x + PR[j].y * w);
			data[cur] = 0xff;
			data[cur + 1] = 0x00;
			data[cur + 2] = 0x00;

			cur = 3*(PT[j].x + PT[j].y * w);
			data[cur] = 0x00;
			data[cur + 1] = 0x00;
			data[cur + 2] = 0xff;

			cur = 3*(PB[j].x + PB[j].y * w);
			data[cur] = 0x00;
			data[cur + 1] = 0x00;
			data[cur + 2] = 0xff;
			//circle(m2, PL[j], 3, Scalar(0, 255, 0), 1);
			//circle(m2, PR[j], 3, Scalar(0, 255, 0), 1);
			//circle(m2, PT[j], 3, Scalar(0, 255, 0), 1);
			//circle(m2, PB[j], 3, Scalar(0, 255, 0), 1);
		}

		imwrite("../../imgs/res3.png", m2);
 	}

	return res;
}


/**
* Creates a BitMatrix by sampling the provided image.
* topLeft, topRight, bottomRight, and bottomLeft are the centers of the squares on the
* diagonal just outside the bull's eye.
*/
static BitMatrix SampleGrid(const BitMatrix& image, const ResultPoint& topLeft, const ResultPoint& topRight, 
	const ResultPoint& bottomRight, const ResultPoint& bottomLeft, bool compact, int nbLayers, int nbCenterLayers, 
	int cx, int cy, Mat mimg)
{
	int dimension = GetDimension(compact, nbLayers);

	float low = dimension / 2.0f - nbCenterLayers;
	float high = dimension / 2.0f + nbCenterLayers;

#if 1
	int w = image.width();
	int h = image.height();
	return get_bits(image, cx, cy, dimension, w, h, mimg);
	/*
	return GridSampler::Instance()->sampleGrid(image,
		dimension,
		dimension,
		low, low,   // topleft
		high, low,  // topright
		high, high, // bottomright
		low, high,  // bottomleft
		topLeft.x(), topLeft.y(),
		topRight.x(), topRight.y(),
		bottomRight.x(), bottomRight.y(),
		bottomLeft.x(), bottomLeft.y());*/
#else
	BitMatrix res(dimension, dimension);
	float w = image.width();
	float h = image.height();

	int left[200], right[200], top[200], bottom[200];
	memset(left, 0x00, sizeof(int) * 200);
	memset(right, 0x00, sizeof(int) * 200);
	memset(top, 0x00, sizeof(int) * 200);
	memset(bottom, 0x00, sizeof(int) * 200);

	//get_differences(image, left, right, top, bottom, dimension);

	// we assume that w == h
	float delta = w / dimension;
	for (int y = 0; y < dimension; y++) {
		for (int x = 0; x < dimension; x++) {
			float dx = (w - left[x] - right[x]) / dimension;
			float dy = (w - top[y] - bottom[y]) / dimension;

			int sy = dy * y + 0.5 + top[y];
			int ey = dy * (y + 1) + 0.5 + top[y];

			int sx = delta * x + 0.5 + left[x];
			int ex = delta * (x + 1) + 0.5 + left[x];
			int sum = 0;
			
			if (ey - sy < 5 || ex - sx < 5) {
				for (int i = sy; i < ey; i++) for (int j = sx; j < ex; j++) {
					bool bBlack = image.get(j, i);
					if (bBlack) sum++;
				}
				int thres = (ey - sy) * (ex - sx) * 5 / 10;
				if (sum > thres)
					res.set(x, y);
			}
			else {

				for (int i = sy + 2; i < ey - 2; i++) for (int j = sx + 2; j < ex - 2; j++) {
					bool bBlack = image.get(j, i);
					if (bBlack) sum++;
				}
				int thres = (ey - sy - 4) * (ex - sx - 4) * 8 / 10;
				if (sum > thres)
					res.set(x, y);
			}
		}
	}

	return res;

#endif
}


DetectorResult Detector::Detect(const BitMatrix& image, bool isMirror)
{
	// checking image 
	int w = image.width();
	int h = image.height();
	Mat img(Size(w, h), CV_8UC3);
	uchar* data = img.data;
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++, data+=3) {
			bool bBlack = image.get(j, i);
			if (bBlack) {
				*data = 0x00;
				*(data + 1) = 0x00;
				*(data + 2) = 0x00;
			}
			else {
				*data = 0xff;
				*(data + 1) = 0xff;
				*(data + 2) = 0xff;
			}
		}
	}
	imshow("bit image", img);
	imwrite("../../imgs/res2.png", img);
	waitKey(0);



	// 1. Get the center of the aztec matrix
	auto pCenter = GetMatrixCenter(image);

	// 2. Get the center points of the four diagonal points just outside the bull's eye
	//  [topRight, bottomRight, bottomLeft, topLeft]
	std::array<ResultPoint, 4> bullsEyeCorners;
	bool compact = false;
	int nbCenterLayers = 0;
	if (!GetBullsEyeCorners(image, pCenter, bullsEyeCorners, compact, nbCenterLayers)) {
		return {};
	}

	if (isMirror) {
		std::swap(bullsEyeCorners[0], bullsEyeCorners[2]);
	}

	// 3. Get the size of the matrix and other parameters from the bull's eye
	int nbLayers = 0;
	int nbDataBlocks = 0;
	int shift = 0;
	if (!ExtractParameters(image, bullsEyeCorners, compact, nbCenterLayers, nbLayers, nbDataBlocks, shift)) {
		return {};
	}

	// 4. Sample the grid
	auto bits = SampleGrid(image, bullsEyeCorners[shift % 4], bullsEyeCorners[(shift + 1) % 4], bullsEyeCorners[(shift + 2) % 4], bullsEyeCorners[(shift + 3) % 4], compact, nbLayers, nbCenterLayers, pCenter.x, pCenter.y, img);
	
	// araj
	if (shift == 2) {
		//these are counter clockwise rotation
		bits.rotate180();
		bits.rotate90();
	}
	else if (shift == 1)
		bits.rotate180();
	else if (shift == 0) {
		bits.rotate90();
	}
	if (bits.empty())
		return {};

	// 5. Get the corners of the matrix.
	GetMatrixCornerPoints(bullsEyeCorners, compact, nbLayers, nbCenterLayers);

	return {std::move(bits), {bullsEyeCorners.begin(), bullsEyeCorners.end()}, compact, nbDataBlocks, nbLayers};
}

} // Aztec
} // ZXing
