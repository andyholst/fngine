// Modified for fngine from cinference:src/serve/request_json.h @ 383e5db. See NOTICE.
#pragma once

#include <nlohmann/json.hpp>

namespace fngine::serve {

// Owning JSON representation for request bodies. Object insertion order is part of the
// model-facing representation whenever a nested object is serialized into PromptInput.
using RequestJson = nlohmann::ordered_json;

} // namespace fngine::serve
