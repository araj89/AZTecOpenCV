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

#include "MultiFormatReader.h"
#include "DecodeHints.h"
#include "BarcodeFormat.h"
#include "Result.h"

//#include "oned/ODReader.h"
//#include "qrcode/QRReader.h"
//#include "datamatrix/DMReader.h"
#include "AZReader.h"
//#include "maxicode/MCReader.h"
//#include "pdf417/PDFReader.h"

#include <memory>

namespace ZXing {

MultiFormatReader::MultiFormatReader(const DecodeHints& hints)
{
	bool tryHarder = hints.tryHarder();
	if (!hints.hasNoFormat()) {
		if (hints.hasFormat(BarcodeFormat::AZTEC)) {
			_readers.emplace_back(new Aztec::Reader());
		}
	}

	if (_readers.empty()) {
		_readers.emplace_back(new Aztec::Reader());
	}
}

MultiFormatReader::~MultiFormatReader() = default;

Result
MultiFormatReader::read(const BinaryBitmap& image) const
{
	// If we have only one reader in our list, just return whatever that decoded.
	// This preserves information (e.g. ChecksumError) instead of just returning 'NotFound'.
	if (_readers.size() == 1)
		return _readers.front()->decode(image);

	for (const auto& reader : _readers) {
		Result r = reader->decode(image);
  		if (r.isValid())
			return r;
	}
	return Result(DecodeStatus::NotFound);
}

} // ZXing