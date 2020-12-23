/*
* Copyright 2017 Huy Cuong Nguyen
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

#include "MultiFormatWriter.h"
#include "BitMatrix.h"
#include "AZWriter.h"
//#include "datamatrix/DMWriter.h"
//#include "pdf417/PDFWriter.h"
//#include "qrcode/QRWriter.h"
//#include "qrcode/QRErrorCorrectionLevel.h"
//#include "oned/ODCodabarWriter.h"
//#include "oned/ODCode39Writer.h"
//#include "oned/ODCode93Writer.h"
//#include "oned/ODCode128Writer.h"
//#include "oned/ODEAN8Writer.h"
//#include "oned/ODEAN13Writer.h"
//#include "oned/ODITFWriter.h"
//#include "oned/ODUPCAWriter.h"
//#include "oned/ODUPCEWriter.h"

namespace ZXing {

namespace {
	
	template <typename Writer, typename EccConverter>
	Writer CreateWriter(CharacterSet encoding, int eccLevel)
	{
		Writer writer;
		if (encoding != CharacterSet::Unknown)
			writer.setEncoding(encoding);
		if (eccLevel >= 0 && eccLevel <= 8)
			writer.setEccPercent(EccConverter()(eccLevel));
		return writer;
	}

	template <typename Writer>
	Writer CreateWriter(int margin)
	{
		Writer writer;
		if (margin >= 0)
			writer.setMargin(margin);
		return writer;
	}

	template <typename Writer, typename EccConverter>
	Writer CreateWriter(CharacterSet encoding, int margin, int eccLevel)
	{
		Writer writer;
		if (encoding != CharacterSet::Unknown)
			writer.setEncoding(encoding);
		if (margin >= 0)
			writer.setMargin(margin);
		if (eccLevel >= 0 && eccLevel <= 8)
			writer.setErrorCorrectionLevel(EccConverter()(eccLevel));
		return writer;
	}

	struct AztecEccConverter {
		int operator()(int eccLevel) const {
			// Aztec supports levels 0 to 100 in percentage
			return eccLevel * 100 / 8;
		}
	};

	struct Pdf417EccConverter {
		int operator()(int eccLevel) const {
			return eccLevel;
		}
	};

	struct QRCodeEccConverter {
		int tmp;
	};

} // anonymous

BitMatrix
MultiFormatWriter::encode(const std::wstring& contents, int width, int height) const
{
	switch (_format)
	{
	case BarcodeFormat::AZTEC:
		return CreateWriter<Aztec::Writer, AztecEccConverter>(_encoding, _eccLevel).encode(contents, width, height);
	default:
		throw std::invalid_argument(std::string("Unsupported format: ") + ToString(_format));
	}
}

} // ZXing
