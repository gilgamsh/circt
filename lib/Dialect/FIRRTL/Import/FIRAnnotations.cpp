//===- FIRAnnotations.cpp - FIRRTL Annotation Utilities -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide utilities related to dealing with FIRRTL Annotations.
//
//===----------------------------------------------------------------------===//

#include "FIRAnnotations.h"
#include "AnnotationDetails.h"

#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationLowering.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

namespace json = llvm::json;

using namespace circt;
using namespace firrtl;
using mlir::UnitAttr;


DictionaryAttr cloneWithNewField(DictionaryAttr anno, StringRef field,
                                 Attribute newValue) {
  auto context = anno.getContext();
  SmallVector<NamedAttribute, 4> newAttr;
  bool found = false;
  for (auto f : anno) {
    if (f.first == field) {
      newAttr.push_back(std::make_pair(f.first, newValue));
      found = true;
    } else {
      newAttr.push_back(f);
    }
  }
  if (!found)
    newAttr.emplace_back(Identifier::get("target", context),
                         StringAttr::get(context, "~"));
  return DictionaryAttr::get(context, newAttr);
}

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
static SmallString<32> canonicalizeTarget(StringRef target) {

  // If this is a normal Target (not a Named), erase that field in the JSON
  // object and return that Target.
  if (target[0] == '~')
    return target;

  // This is a legacy target using the firrtl.annotations.Named type.  This
  // can be trivially canonicalized to a non-legacy target, so we do it with
  // the following three mappings:
  //   1. CircuitName => CircuitTarget, e.g., A -> ~A
  //   2. ModuleName => ModuleTarget, e.g., A.B -> ~A|B
  //   3. ComponentName => ReferenceTarget, e.g., A.B.C -> ~A|B>C
  SmallString<32> newTarget("~");
  unsigned tokenIdx = 0;
  for (auto a : target) {
    if (a == '.') {
      switch (tokenIdx) {
      case 0:
        newTarget += '|';
        break;
      case 1:
        newTarget += '>';
        break;
      default:
        newTarget += '\'';
        break;
      }
      ++tokenIdx;
    } else
      newTarget += a;
  }
  return newTarget;
}
static StringAttr canonicalizeTarget(StringAttr target) {

  // If this is a normal Target (not a Named), erase that field in the JSON
  // object and return that Target.
  if (target.getValue()[0] == '~')
    return target;

  return StringAttr::get(target.getContext(),
                         canonicalizeTarget(target.getValue()));
}

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the value
/// of a specific type associated with a key in a dictionary.  However, this is
/// specialized to print a useful error message, specific to custom annotation
/// process, on failure.
template <typename A>
static A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
                  Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = value.dyn_cast_or_null<A>();
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

/// Convert arbitrary JSON to an MLIR Attribute.
static Attribute convertJSONToAttribute(MLIRContext *context,
                                        json::Value &value, json::Path p) {
  // String or quoted JSON
  if (auto a = value.getAsString()) {
    // Test to see if this might be quoted JSON (a string that is actually
    // JSON).  Sometimes FIRRTL developers will do this to serialize objects
    // that the Scala FIRRTL Compiler doesn't know about.
    auto unquotedValue = json::parse(a.getValue());
    auto err = unquotedValue.takeError();
    // If this parsed without an error, then it's more JSON and recurse on
    // that.
    if (!err)
      return convertJSONToAttribute(context, unquotedValue.get(), p);
    // If there was an error, then swallow it and handle this as a string.
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {});
    return StringAttr::get(context, a.getValue());
  }

  // Integer
  if (auto a = value.getAsInteger())
    return IntegerAttr::get(IntegerType::get(context, 64), a.getValue());

  // Float
  if (auto a = value.getAsNumber())
    return FloatAttr::get(mlir::FloatType::getF64(context), a.getValue());

  // Boolean
  if (auto a = value.getAsBoolean())
    return BoolAttr::get(context, a.getValue());

  // Null
  if (auto a = value.getAsNull())
    return mlir::UnitAttr::get(context);

  // Object
  if (auto a = value.getAsObject()) {
    NamedAttrList metadata;
    for (auto b : *a)
      metadata.append(
          b.first, convertJSONToAttribute(context, b.second, p.field(b.first)));
    return DictionaryAttr::get(context, metadata);
  }

  // Array
  if (auto a = value.getAsArray()) {
    SmallVector<Attribute> metadata;
    for (size_t i = 0, e = (*a).size(); i != e; ++i)
      metadata.push_back(convertJSONToAttribute(context, (*a)[i], p.index(i)));
    return ArrayAttr::get(context, metadata);
  }

  llvm_unreachable("Impossible unhandled JSON type");
}

static std::string
addNLATargets(MLIRContext *context, StringRef targetStrRef, CircuitOp circuit,
              size_t &nlaNumber, NamedAttrList &metadata,
              SmallVectorImpl<Attribute> &mutableAnnotationMap) {

  auto nlaTargets = expandNonLocal(targetStrRef);

  FlatSymbolRefAttr nlaSym;
  if (nlaTargets.size() > 1) {
    nlaSym = buildNLA(circuit, ++nlaNumber, nlaTargets);
    metadata.append("circt.nonlocal", nlaSym);
  }

  for (int i = 0, e = nlaTargets.size() - 1; i < e; ++i) {
    NamedAttrList pathmetadata;
    pathmetadata.append("circt.nonlocal", nlaSym);
    pathmetadata.append("class", StringAttr::get(context, "circt.nonlocal"));
    pathmetadata.append("target",
                        StringAttr::get(context, std::get<0>(nlaTargets[i])));
    mutableAnnotationMap.push_back(DictionaryAttr::get(context, pathmetadata));
  }

  // Annotations on the element instance.
  auto leafTarget =
      splitAndAppendTarget(metadata, std::get<0>(nlaTargets.back()), context)
          .first;

  return leafTarget.str();
}

/// Examine an Annotation JSON object and return an optional string indicating
/// the target associated with this annotation.  Automatically convert any
/// legacy Named targets to actual Targets.  Note: it is expected that a target
/// may not exist, e.g., any subclass of firrtl.annotations.NoTargetAnnotation
/// will not have a target.
DictionaryAttr firrtl::normalizeTarget(DictionaryAttr anno) {

  auto maybeTarget = anno.getNamed("target");
  if (maybeTarget) {
    auto targetStr = maybeTarget->second.dyn_cast<StringAttr>();
    if (targetStr) {
      auto newTarget = canonicalizeTarget(targetStr);
      if (newTarget == targetStr)
        return anno;
      return cloneWithNewField(anno, "target", newTarget);
    } else {
      // p.field("target").report("target must be a string type");
      return {};
    }
  } else {
    return cloneWithNewField(anno, "target",
                             StringAttr::get(anno.getContext(), "~"));
  }
};

//     // Build a mutable map of Target to Annotation.
//   llvm::StringMap<llvm::SmallVector<Attribute>> mutableAnnotationMap;

//   for (size_t i = 0, e = (*array).size(); i != e; ++i) {
//     auto object = (*array)[i].getAsObject();
//     auto p = path.index(i);
//     if (!object) {
//       p.report("Expected annotations to be an array of objects, but found an
//       "
//                "array of something else.");
//       return false;
//     }
//     // Find and remove the "target" field from the Annotation object if it
//     // exists.  In the FIRRTL Dialect, the target will be implicitly
//     specified
//     // based on where the attribute is applied.
//     auto optTarget = findAndEraseTarget(object, p);
//     if (!optTarget)
//       return false;
//     StringRef targetStrRef = optTarget.getValue();

//     if (targetStrRef != "~") {
//       auto circuitFieldEnd = targetStrRef.find_first_of('|');
//       if (circuitTarget != targetStrRef.take_front(circuitFieldEnd)) {
//         p.report("annotation has invalid circuit name");
//         return false;
//       }
//     }

//     // Build up the Attribute to represent the Annotation and store it in the
//     // global Target -> Attribute mapping.
//     NamedAttrList metadata;
//     for (auto field : *object) {
//       if (auto value = convertJSONToAttribute(context, field.second, p)) {
//         metadata.append(field.first, value);
//         continue;
//       }
//       return false;
//     }

//     auto leafTarget = addNLATargets(context, targetStrRef, circuit,
//     nlaNumber,
//                                     metadata, mutableAnnotationMap);

//     mutableAnnotationMap[leafTarget].push_back(
//         DictionaryAttr::get(context, metadata));
//   }

//   // Convert the mutable Annotation map to a SmallVector<ArrayAttr>.
//   for (auto a : mutableAnnotationMap.keys()) {
//     // If multiple annotations on a single object, then append it.
//     if (annotationMap.count(a))
//       for (auto attr : annotationMap[a])
//         mutableAnnotationMap[a].push_back(attr);

//     annotationMap[a] = ArrayAttr::get(context, mutableAnnotationMap[a]);
//   }

//   return true;
// }

/// Convert a JSON value containing OMIR JSON (an array of OMNodes), convert
/// this to an OMIRAnnotation, and add it to a mutable `annotationMap` argument.
bool circt::firrtl::fromOMIRJSON(json::Value &value,
                                 SmallVectorImpl<Attribute> &attrs,
                                 json::Path path, MLIRContext *context) {
  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto *array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected OMIR to be an array of nodes, but found something else.");
    return false;
  }

  // Build a mutable map of Target to Annotation.
  SmallVector<Attribute> omnodes;
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto *object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected OMIR to be an array of objects, but found an array of "
               "something else.");
      return false;
    }

    // Manually built up OMNode.
    NamedAttrList omnode;

    // Validate that this looks like an OMNode.  This should have three fields:
    //   - "info": String
    //   - "id": String that starts with "OMID:"
    //   - "fields": Array<Object>
    // Fields is optional and is a dictionary encoded as an array of objects:
    //   - "info": String
    //   - "name": String
    //   - "value": JSON
    // The dictionary is keyed by the "name" member and the array of fields is
    // guaranteed to not have collisions of the "name" key.
    auto maybeInfo = object->getString("info");
    if (!maybeInfo) {
      p.report("OMNode missing mandatory member \"info\" with type \"string\"");
      return false;
    }
    auto maybeID = object->getString("id");
    if (!maybeID || !maybeID.getValue().startswith("OMID:")) {
      p.report("OMNode missing mandatory member \"id\" with type \"string\" "
               "that starts with \"OMID:\"");
      return false;
    }
    auto *maybeFields = object->get("fields");
    if (maybeFields && !maybeFields->getAsArray()) {
      p.report("OMNode has \"fields\" member with incorrect type (expected "
               "\"array\")");
      return false;
    }
    Attribute fields;
    if (!maybeFields)
      fields = DictionaryAttr::get(context, {});
    else {
      auto array = *maybeFields->getAsArray();
      NamedAttrList fieldAttrs;
      for (size_t i = 0, e = array.size(); i != e; ++i) {
        auto *field = array[i].getAsObject();
        auto pI = p.field("fields").index(i);
        if (!field) {
          pI.report("OMNode has field that is not an \"object\"");
          return false;
        }
        auto maybeInfo = field->getString("info");
        if (!maybeInfo) {
          pI.report(
              "OMField missing mandatory member \"info\" with type \"string\"");
          return false;
        }
        auto maybeName = field->getString("name");
        if (!maybeName) {
          pI.report(
              "OMField missing mandatory member \"name\" with type \"string\"");
          return false;
        }
        auto *maybeValue = field->get("value");
        if (!maybeValue) {
          pI.report("OMField missing mandatory member \"value\"");
          return false;
        }
        NamedAttrList values;
        values.append("info", StringAttr::get(context, maybeInfo.getValue()));
        values.append("value", convertJSONToAttribute(context, *maybeValue,
                                                      pI.field("value")));
        fieldAttrs.append(maybeName.getValue(),
                          DictionaryAttr::get(context, values));
      }
      fields = DictionaryAttr::get(context, fieldAttrs);
    }

    omnode.append("info", StringAttr::get(context, maybeInfo.getValue()));
    omnode.append("id", convertJSONToAttribute(context, *object->get("id"),
                                               p.field("id")));
    omnode.append("fields", fields);
    attrs.push_back(DictionaryAttr::get(context, omnode));
  }

  return true;
}

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// any annotations it may contain.  This is going to generate two types of
/// annotations:
///   1) Annotations necessary to build interfaces and store them at "~"
///   2) Scattered annotations for how components bind to interfaces
static Optional<DictionaryAttr> parseAugmentedType(
    MLIRContext *context, DictionaryAttr augmentedType, DictionaryAttr root,
    SmallVectorImpl<Attribute> &newAnnotations, StringRef companion,
    StringAttr name, StringAttr defName, Optional<IntegerAttr> id,
    Optional<StringAttr>(description), Location loc, unsigned &annotationID,
    Twine clazz, Twine path = {}) {

  /// Return a new identifier that can be used to link scattered annotations
  /// together.  This mutates the by-reference parameter annotationID.
  auto newID = [&]() {
    return IntegerAttr::get(IntegerType::get(context, 64), annotationID++);
  };

  /// Optionally unpack a ReferenceTarget encoded as a DictionaryAttr.  Return
  /// either a pair containing the Target string (up to the reference) and an
  /// array of components or none if the input is malformed.  The input
  /// DicionaryAttr encoding is a JSON object of a serialized ReferenceTarget
  /// Scala class.  By example, this is converting:
  ///   ~Foo|Foo>a.b[0]
  /// To:
  ///   {"~Foo|Foo>a", {".b", "[0]"}}
  /// The format of a ReferenceTarget object like:
  ///   circuit: String
  ///   module: String
  ///   path: Seq[(Instance, OfModule)]
  ///   ref: String
  ///   component: Seq[TargetToken]
  auto refTargetToString = [&](DictionaryAttr refTarget)
      -> llvm::Optional<std::pair<std::string, ArrayAttr>> {
    auto circuitAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "circuit", loc, clazz, path);
    auto moduleAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "module", loc, clazz, path);
    auto pathAttr =
        tryGetAs<ArrayAttr>(refTarget, refTarget, "path", loc, clazz, path);
    auto componentAttr = tryGetAs<ArrayAttr>(refTarget, refTarget, "component",
                                             loc, clazz, path);
    if (!circuitAttr || !moduleAttr || !pathAttr || !componentAttr)
      return llvm::Optional<std::pair<std::string, ArrayAttr>>();

    // Parse non-local annotations.
    SmallString<32> strpath;
    for (auto p : pathAttr) {
      auto dict = p.dyn_cast_or_null<DictionaryAttr>();
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto instHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_1", loc, clazz, path);
      auto modHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_2", loc, clazz, path);
      if (!instHolder || !modHolder) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto inst = tryGetAs<StringAttr>(instHolder, instHolder, "value", loc,
                                       clazz, path);
      auto mod =
          tryGetAs<StringAttr>(modHolder, modHolder, "value", loc, clazz, path);
      if (!inst || !mod) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      strpath += "/" + inst.getValue().str() + ":" + mod.getValue().str();
    }

    auto refAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "ref", loc, clazz, path);
    SmallVector<Attribute> componentAttrs;
    for (size_t i = 0, e = componentAttr.size(); i != e; ++i) {
      auto cPath = (path + ".component[" + Twine(i) + "]").str();
      auto component = componentAttr[i];
      auto dict = component.dyn_cast_or_null<DictionaryAttr>();
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz + "' with path '" + cPath +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto classAttr =
          tryGetAs<StringAttr>(dict, refTarget, "class", loc, clazz, cPath);
      if (!classAttr)
        return {};

      auto value = dict.get("value");

      // A subfield like "bar" in "~Foo|Foo>foo.bar".
      if (auto field = value.dyn_cast<StringAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Field" &&
               "A StringAttr target token must be found with a subfield target "
               "token.");
        componentAttrs.push_back(field);
        continue;
      }

      // A subindex like "42" in "~Foo|Foo>foo[42]".
      if (auto index = value.dyn_cast<IntegerAttr>()) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Index" &&
               "An IntegerAttr target token must be found with a subindex "
               "target token.");
        componentAttrs.push_back(index);
        continue;
      }

      mlir::emitError(loc,
                      "Annotation '" + clazz + "' with path '" + cPath +
                          ".value has unexpected type (should be StringAttr "
                          "for subfield  or IntegerAttr for subindex).")
              .attachNote()
          << "The value received was: " << value << "\n";
      return {};
    }

    return llvm::Optional<std::pair<std::string, ArrayAttr>>(
        {(Twine("~" + circuitAttr.getValue() + "|" + moduleAttr.getValue() +
                strpath + ">" + refAttr.getValue()))
             .str(),
         ArrayAttr::get(context, componentAttrs)});
  };

  auto classAttr =
      tryGetAs<StringAttr>(augmentedType, root, "class", loc, clazz, path);
  if (!classAttr)
    return None;
  StringRef classBase = classAttr.getValue();
  if (!classBase.consume_front("sifive.enterprise.grandcentral.Augmented")) {
    mlir::emitError(loc,
                    "the 'class' was expected to start with "
                    "'sifive.enterprise.grandCentral.Augmented*', but was '" +
                        classAttr.getValue() + "' (Did you misspell it?)")
            .attachNote()
        << "see annotation: " << augmentedType;
    return None;
  }

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classBase == "BundleType") {
    defName =
        tryGetAs<StringAttr>(augmentedType, root, "defName", loc, clazz, path);
    if (!defName)
      return None;

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmenetedType
    SmallVector<Attribute> elements;
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    for (size_t i = 0, e = elementsAttr.size(); i != e; ++i) {
      auto field = elementsAttr[i].dyn_cast_or_null<DictionaryAttr>();
      if (!field) {
        mlir::emitError(
            loc,
            "Annotation '" + Twine(clazz) + "' with path '.elements[" +
                Twine(i) +
                "]' contained an unexpected type (expected a DictionaryAttr).")
                .attachNote()
            << "The received element was: " << elementsAttr[i] << "\n";
        return None;
      }
      auto ePath = (path + ".elements[" + Twine(i) + "]").str();
      auto name = tryGetAs<StringAttr>(field, root, "name", loc, clazz, ePath);
      auto tpe =
          tryGetAs<DictionaryAttr>(field, root, "tpe", loc, clazz, ePath);
      Optional<StringAttr> description = None;
      if (auto maybeDescription = field.get("description"))
        description = maybeDescription.cast<StringAttr>();
      auto eltAttr = parseAugmentedType(
          context, tpe, root, newAnnotations, companion, name, defName, None,
          description, loc, annotationID, clazz, path);
      if (!name || !tpe || !eltAttr)
        return None;

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", maybeDescription.cast<StringAttr>());
      attrs.append("name", name);
      attrs.append("tpe", tpe.getAs<StringAttr>("class"));
      elements.push_back(eltAttr.getValue());
    }
    // Add an annotation that stores information necessary to construct the
    // module for the view.  This needs the name of the module (defName) and the
    // names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    if (id)
      attrs.append("id", id.getValue());
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // An AugmentedGroundType looks like:
  //   "ref": ReferenceTarget
  //   "tpe": GroundType
  // The ReferenceTarget is not serialized to a string.  The GroundType will
  // either be an actual FIRRTL ground type or a GrandCentral uninferred type.
  // This can be ignored for us.
  if (classBase == "GroundType") {
    auto maybeTarget =
        refTargetToString(augmentedType.getAs<DictionaryAttr>("ref"));
    if (!maybeTarget) {
      mlir::emitError(loc, "Failed to parse ReferenceTarget").attachNote()
          << "See the full Annotation here: " << root;
      return None;
    }

    auto id = newID();

    // TODO: We don't support non-local annotations, so force this annotation
    // into a local annotation.  This does not properly check that the
    // non-local and local targets are totally equivalent.
    auto target = maybeTarget.getValue();
    auto localTarget = std::get<0>(expandNonLocal(target.first).back());
    auto subTargets = target.second;

    NamedAttrList elementIface, elementScattered, dontTouch;

    // Populate the annotation for the interface element.
    elementIface.append("class", classAttr);
    if (description)
      elementIface.append("description", description.getValue());
    elementIface.append("id", id);
    elementIface.append("name", name);
    // Populate an annotation that will be scattered onto the element.
    elementScattered.append("class", classAttr);
    elementScattered.append("id", id);
    // Populate a dont touch annotation for the scattered element.
    dontTouch.append(
        "class",
        StringAttr::get(context, "firrtl.transforms.DontTouchAnnotation"));
    // If there are sub-targets, then add these.
    if (subTargets) {
      elementScattered.append("target", subTargets);
      dontTouch.append("target", subTargets);
    }
    elementScattered.append("target", StringAttr::get(context, localTarget));
    newAnnotations.push_back(
        DictionaryAttr::getWithSorted(context, elementScattered));
    dontTouch.append("target", StringAttr::get(context, localTarget));
    newAnnotations.push_back(DictionaryAttr::getWithSorted(context, dontTouch));

    return DictionaryAttr::getWithSorted(context, elementIface);
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classBase == "VectorType") {
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return None;
    SmallVector<Attribute> elements;
    for (auto elt : elementsAttr) {
      auto eltAttr = parseAugmentedType(context, elt.cast<DictionaryAttr>(),
                                        root, newAnnotations, companion, name,
                                        StringAttr::get(context, ""), id, None,
                                        loc, annotationID, clazz, path);
      if (!eltAttr)
        return None;
      elements.push_back(eltAttr.getValue());
    }
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    if (description)
      attrs.append("description", description.getValue());
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Any of the following are known and expected, but are legacy AugmentedTypes
  // do not have a target:
  //   - AugmentedStringType
  //   - AugmentedBooleanType
  //   - AugmentedIntegerType
  //   - AugmentedDoubleType
  bool isIgnorable =
      llvm::StringSwitch<bool>(classBase)
          .Cases("StringType", "BooleanType", "IntegerType", "DoubleType", true)
          .Default(false);
  if (isIgnorable)
    return augmentedType;

  // Anything else is unexpected or a user error if they manually wrote
  // annotations.  Print an error and error out.
  mlir::emitError(loc, "found unknown AugmentedType '" + classAttr.getValue() +
                           "' (Did you misspell it?)")
          .attachNote()
      << "see annotation: " << augmentedType;
  return None;
}

/// Recursively walk Object Model IR and convert FIRRTL targets to identifiers
/// while scattering trackers into the newAnnotations argument.
///
/// Object Model IR consists of a type hierarchy built around recursive arrays
/// and dictionaries whose leaves are "string-encoded types".  This is an Object
/// Model-specific construct that puts type information alongside a value.
/// Concretely, these look like:
///
///     'OM' type ':' value
///
/// This function is only concerned with unpacking types whose values are FIRRTL
/// targets.  This is because these need to be kept up-to-date with
/// modifications made to the circuit whereas other types are just passing
/// through CIRCT.
///
/// At a later time this understanding may be expanded or Object Model IR may
/// become its own Dialect.  At this time, this function is trying to do as
/// minimal work as possible to just validate that the OMIR looks okay without
/// doing lots of unnecessary unpacking/repacking of string-encoded types.
static Optional<Attribute>
scatterOMIR(Attribute original, unsigned &annotationID,
            SmallVectorImpl<Attribute> &newAnnotations, CircuitOp circuit,
            size_t &nlaNumber) {
  auto *ctx = original.getContext();

  // Convert a string-encoded type to a dictionary that includes the type
  // information and an identifier derived from the current annotationID.  Then
  // increment the annotationID.  Return the constructed dictionary.
  auto addID = [&](StringRef tpe, StringRef path) -> DictionaryAttr {
    NamedAttrList fields;
    fields.append("id",
                  IntegerAttr::get(IntegerType::get(ctx, 64), annotationID++));
    fields.append("omir.tracker", UnitAttr::get(ctx));
    fields.append("path", StringAttr::get(ctx, path));
    fields.append("type", StringAttr::get(ctx, tpe));
    return DictionaryAttr::getWithSorted(ctx, fields);
  };

  return TypeSwitch<Attribute, Optional<Attribute>>(original)
      // Most strings in the Object Model are actually string-encoded types.
      // These are types which look like: "<type>:<value>".  This code will
      // examine all strings, parse them into type and value, and then either
      // store them in their unpacked state (and possibly scatter trackers into
      // the circuit), store them in their packed state (because CIRCT is not
      // expected to care about them right now), or error if we see them
      // (because they should not exist and are expected to serialize to a
      // different format).
      .Case<StringAttr>([&](StringAttr str) -> Optional<Attribute> {
        // Unpack the string into type and value.
        StringRef tpe, value;
        std::tie(tpe, value) = str.getValue().split(":");

        // These are string-encoded types that are targets in the circuit.
        // These require annotations to be scattered for them.  Replace their
        // target with an ID and scatter a tracker.
        if (tpe == "OMReferenceTarget" || tpe == "OMMemberReferenceTarget" ||
            tpe == "OMMemberInstanceTarget" || tpe == "OMInstanceTarget" ||
            tpe == "OMDontTouchedReferenceTarget") {
          NamedAttrList tracker;
          tracker.append("class", StringAttr::get(ctx, omirTrackerAnnoClass));
          tracker.append(
              "id", IntegerAttr::get(IntegerType::get(ctx, 64), annotationID));

          auto canonTarget = canonicalizeTarget(value);

          auto leafTarget = addNLATargets(ctx, canonTarget, circuit, nlaNumber,
                                          tracker, newAnnotations);
          tracker.append("target", StringAttr::get(ctx, leafTarget));
          newAnnotations.push_back(DictionaryAttr::get(ctx, tracker));

          return addID(tpe, value);
        }

        // The following are types that may exist, but we do not unbox them.  At
        // a later time, we may want to change this behavior and unbox these if
        // we wind up building out an Object Model dialect:
        if (isOMIRStringEncodedPassthrough(tpe))
          return str;

        // The following types are not expected to exist because they have
        // serializations to JSON types or are removed during serialization.
        // Hence, any of the following types are NOT expected to exist and we
        // error if we see them.  These are explicitly specified as opposed to
        // being handled in the "unknown" catch-all case below because we want
        // to provide a good error message that a user may be doing something
        // very weird.
        if (tpe == "OMMap" || tpe == "OMArray" || tpe == "OMBoolean" ||
            tpe == "OMInt" || tpe == "OMDouble" || tpe == "OMFrozenTarget") {
          auto diag =
              mlir::emitError(circuit.getLoc())
              << "found known string-encoded OMIR type \"" << tpe
              << "\", but this type should not be seen as it has a defined "
                 "serialization format that does NOT use a string-encoded type";
          diag.attachNote()
              << "the problematic OMIR is reproduced here: " << original;
          return None;
        }

        // This is a catch-all for any unknown types.
        auto diag = mlir::emitError(circuit.getLoc())
                    << "found unknown string-encoded OMIR type \"" << tpe
                    << "\" (Did you misspell it?  Is CIRCT missing an Object "
                       "Model OMIR type?)";
        diag.attachNote() << "the problematic OMIR is reproduced here: "
                          << original;
        return None;
      })
      // For an array, just recurse into each element and rewrite the array with
      // the results.
      .Case<ArrayAttr>([&](ArrayAttr arr) -> Optional<Attribute> {
        SmallVector<Attribute> newArr;
        for (auto element : arr) {
          auto newElement = scatterOMIR(element, annotationID, newAnnotations,
                                        circuit, nlaNumber);
          if (!newElement)
            return None;
          newArr.push_back(newElement.getValue());
        }
        return ArrayAttr::get(ctx, newArr);
      })
      // For a dictionary, recurse into each value and rewrite the key/value
      // pairs.
      .Case<DictionaryAttr>([&](DictionaryAttr dict) -> Optional<Attribute> {
        NamedAttrList newAttrs;
        for (auto pairs : dict) {
          auto maybeValue = scatterOMIR(pairs.second, annotationID,
                                        newAnnotations, circuit, nlaNumber);
          if (!maybeValue)
            return None;
          newAttrs.append(pairs.first, maybeValue.getValue());
        }
        return DictionaryAttr::get(ctx, newAttrs);
      })
      // These attributes are all expected.  They are OMIR types, but do not
      // have string-encodings (hence why these should error if we see them as
      // strings).
      .Case</* OMBoolean */ BoolAttr, /* OMDouble */ FloatAttr,
            /* OMInt */ IntegerAttr>(
          [](auto passThrough) { return passThrough; })
      // Error if we see anything else.
      .Default([&](auto) -> Optional<Attribute> {
        auto diag = mlir::emitError(circuit.getLoc())
                    << "found unexpected MLIR attribute \"" << original
                    << "\" while trying to scatter OMIR";
        return None;
      });
}

/// Convert an Object Model Field into an optional pair of a string key and a
/// dictionary attribute.  Expand internal source locator strings to location
/// attributes.  Scatter any FIRRTL targets into the circuit. If this is an
/// illegal Object Model Field return None.
///
/// Each Object Model Field consists of three mandatory members with
/// the following names and types:
///
///   - "info": Source Locator String
///   - "name": String
///   - "value": Object Model IR
///
/// The key is the "name" and the dictionary consists of the "info" and "value"
/// members.  Each value is recursively traversed to scatter any FIRRTL targets
/// that may be used inside it.
///
/// This conversion from an object (dictionary) to key--value pair is safe
/// because each Object Model Field in an Object Model Node must have a unique
/// "name".  Anything else is illegal Object Model.
static Optional<std::pair<StringRef, DictionaryAttr>>
scatterOMField(Attribute original, const Attribute root, unsigned &annotationID,
               SmallVectorImpl<Attribute> &newAnnotations, CircuitOp circuit,
               size_t &nlaNumber, Location loc, unsigned index) {
  // The input attribute must be a dictionary.
  DictionaryAttr dict = original.dyn_cast<DictionaryAttr>();
  if (!dict) {
    llvm::errs() << "OMField is not a dictionary, but should be: " << original
                 << "\n";
    return None;
  }

  auto *ctx = circuit.getContext();

  // Generate an arbitrary identifier to use for caching when using
  // `maybeStringToLocation`.
  Identifier locatorFilenameCache = Identifier::get(".", ctx);
  FileLineColLoc fileLineColLocCache;

  // Convert location from a string to a location attribute.
  auto infoAttr = tryGetAs<StringAttr>(dict, root, "info", loc, omirAnnoClass);
  if (!infoAttr)
    return None;
  auto maybeLoc =
      maybeStringToLocation(infoAttr.getValue(), false, locatorFilenameCache,
                            fileLineColLocCache, ctx);
  mlir::LocationAttr infoLoc;
  if (maybeLoc.first)
    infoLoc = maybeLoc.second.getValue();
  else
    infoLoc = UnknownLoc::get(ctx);

  // Extract the name attribute.
  auto nameAttr = tryGetAs<StringAttr>(dict, root, "name", loc, omirAnnoClass);
  if (!nameAttr)
    return None;

  // The value attribute is unstructured and just copied over.
  auto valueAttr = tryGetAs<Attribute>(dict, root, "value", loc, omirAnnoClass);
  if (!valueAttr)
    return None;
  auto newValue =
      scatterOMIR(valueAttr, annotationID, newAnnotations, circuit, nlaNumber);
  if (!newValue)
    return None;

  NamedAttrList values;
  // We add the index if one was provided.  This can be used later to
  // reconstruct the order of the original array.
  values.append("index", IntegerAttr::get(IntegerType::get(ctx, 64), index));
  values.append("info", infoLoc);
  values.append("value", newValue.getValue());

  return {{nameAttr.getValue(), DictionaryAttr::getWithSorted(ctx, values)}};
}

/// Convert an Object Model Node to an optional dictionary, convert source
/// locator strings to location attributes, and scatter FIRRTL targets into the
/// circuit.  If this is an illegal Object Model Node, then return None.
///
/// An Object Model Node is expected to look like:
///
///   - "info": Source Locator String
///   - "id": String-encoded integer ('OMID' ':' Integer)
///   - "fields": Array<Object>
///
/// The "fields" member may be absent.  If so, then construct an empty array.
static Optional<DictionaryAttr>
scatterOMNode(Attribute original, const Attribute root, unsigned &annotationID,
              SmallVectorImpl<Attribute> &newAnnotations, CircuitOp circuit,
              size_t &nlaNumber, Location loc) {

  /// The input attribute must be a dictionary.
  DictionaryAttr dict = original.dyn_cast<DictionaryAttr>();
  if (!dict) {
    llvm::errs() << "OMNode is not a dictionary, but should be: " << original
                 << "\n";
    return None;
  }

  NamedAttrList omnode;
  auto *ctx = circuit.getContext();

  // Generate an arbitrary identifier to use for caching when using
  // `maybeStringToLocation`.
  Identifier locatorFilenameCache = Identifier::get(".", ctx);
  FileLineColLoc fileLineColLocCache;

  // Convert the location from a string to a location attribute.
  auto infoAttr = tryGetAs<StringAttr>(dict, root, "info", loc, omirAnnoClass);
  if (!infoAttr)
    return None;
  auto maybeLoc =
      maybeStringToLocation(infoAttr.getValue(), false, locatorFilenameCache,
                            fileLineColLocCache, ctx);
  mlir::LocationAttr infoLoc;
  if (maybeLoc.first)
    infoLoc = maybeLoc.second.getValue();
  else
    infoLoc = UnknownLoc::get(ctx);

  // Extract the OMID.  Don't parse this, just leave it as a string.
  auto idAttr = tryGetAs<StringAttr>(dict, root, "id", loc, omirAnnoClass);
  if (!idAttr)
    return None;

  // Convert the fields from an ArrayAttr to a DictionaryAttr keyed by their
  // "name".  If no fields member exists, then just create an empty dictionary.
  // Note that this is safe to construct because all fields must have unique
  // "name" members relative to each other.
  auto maybeFields = dict.getAs<ArrayAttr>("fields");
  DictionaryAttr fields;
  if (!maybeFields)
    fields = DictionaryAttr::get(ctx);
  else {
    auto fieldAttr = maybeFields.getValue();
    NamedAttrList fieldAttrs;
    for (size_t i = 0, e = fieldAttr.size(); i != e; ++i) {
      auto field = fieldAttr[i];
      if (auto newField =
              scatterOMField(field, root, annotationID, newAnnotations, circuit,
                             nlaNumber, loc, i)) {
        fieldAttrs.append(newField.getValue().first,
                          newField.getValue().second);
        continue;
      }
      return None;
    }
    fields = DictionaryAttr::get(ctx, fieldAttrs);
  }

  omnode.append("fields", fields);
  omnode.append("id", idAttr);
  omnode.append("info", infoLoc);

  return DictionaryAttr::getWithSorted(ctx, omnode);
}

/// Main entry point to handle scattering of an OMIRAnnotation.  Return the
/// modified optional attribute on success and None on failure.  Any scattered
/// annotations will be added to the reference argument `newAnnotations`.
static Optional<DictionaryAttr>
scatterOMIRAnnotation(DictionaryAttr dict, unsigned &annotationID,
                      SmallVectorImpl<Attribute> &newAnnotations,
                      CircuitOp circuit, size_t &nlaNumber, Location loc) {

  auto nodes = tryGetAs<ArrayAttr>(dict, dict, "nodes", loc, omirAnnoClass);
  if (!nodes)
    return None;

  SmallVector<Attribute> newNodes;
  for (auto node : nodes) {
    auto newNode = scatterOMNode(node, dict, annotationID, newAnnotations,
                                 circuit, nlaNumber, loc);
    if (!newNode)
      return None;
    newNodes.push_back(newNode.getValue());
  }

  auto *ctx = circuit.getContext();

  NamedAttrList newAnnotation;
  newAnnotation.append("class", StringAttr::get(ctx, omirAnnoClass));
  newAnnotation.append("nodes", ArrayAttr::get(ctx, newNodes));
  return DictionaryAttr::get(ctx, newAnnotation);
}

/// Deserialize a JSON value into FIRRTL Annotations.  Annotations are
/// represented as a Target-keyed arrays of attributes.  The input JSON value is
/// checked, at runtime, to be an array of objects.  Returns true if successful,
/// false if unsuccessful.
bool circt::firrtl::fromJSON(json::Value &value,
                             SmallVectorImpl<Attribute> &attrs, json::Path path,
                             MLIRContext *context) {

  // The JSON value must be an array of objects.  Anything else is reported as
  // invalid.
  auto array = value.getAsArray();
  if (!array) {
    path.report(
        "Expected annotations to be an array, but found something else.");
    return false;
  }

  // Build an array of annotations.
  for (size_t i = 0, e = (*array).size(); i != e; ++i) {
    auto object = (*array)[i].getAsObject();
    auto p = path.index(i);
    if (!object) {
      p.report("Expected annotations to be an array of objects, but found an "
               "array of something else.");
      return false;
    }

    // Build up the Attribute to represent the Annotation
    NamedAttrList metadata;

    for (auto field : *object) {
      if (auto value = convertJSONToAttribute(context, field.second, p)) {
        metadata.append(field.first, value);
        continue;
      }
      return false;
    }

    attrs.push_back(DictionaryAttr::get(context, metadata));
  }

  return true;
}
