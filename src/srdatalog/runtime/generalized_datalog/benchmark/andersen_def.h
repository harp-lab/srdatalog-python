#pragma once

#include "ast.h"
#include "semiring.h"

namespace sd = SRDatalog;
namespace ast = sd::AST;

using ast::database;
using ast::fixpoint;
using ast::non_iterative;
using ast::rel;
using ast::relation;
using ast::operator""_v;
using ast::Literals::operator""_s;

using SR = BooleanSR;

// Define schemas for Andersen's Points-To Analysis
using AddressOfSchema = relation<decltype("AddressOf"_s), SR, int, int>;
using AssignSchema = relation<decltype("Assign"_s), SR, int, int>;
using LoadSchema = relation<decltype("Load"_s), SR, int, int>;
using StoreSchema = relation<decltype("Store"_s), SR, int, int>;
using PointsToSchema = relation<decltype("PointsTo"_s), SR, int, int>;

using AndersenBlueprint =
    database<AddressOfSchema, AssignSchema, LoadSchema, StoreSchema, PointsToSchema>;

// Variables
constexpr auto x_ = "x"_v;
constexpr auto y_ = "y"_v;
constexpr auto z_ = "z"_v;
constexpr auto w_ = "w"_v;

// Relation accessors
constexpr auto addressOf = rel<AddressOfSchema>;
constexpr auto assign = rel<AssignSchema>;
constexpr auto load = rel<LoadSchema>;
constexpr auto store = rel<StoreSchema>;
constexpr auto pointsTo = rel<PointsToSchema>;

// Base rule (non-iterative SCC): PointsTo(y, x) :- AddressOf(y, x)
const auto andersen_base_rule = pointsTo.newt(y_, x_) <<= addressOf.full(y_, x_);

// Rule 1: PointsTo(y, x) :- Assign(y, z), PointsTo(z, x)
const auto assign_rule = pointsTo.newt(y_, x_) <<= (pointsTo.delta(z_, x_), assign.full(y_, z_));

// Rule 2: PointsTo(y, w) :- Load(y, x), PointsTo(x, z), PointsTo(z, w)
const auto load_variant1 =
    (pointsTo.newt(y_, w_) <<= (pointsTo.delta(x_, z_), load.full(y_, x_), pointsTo.full(z_, w_)));
const auto load_variant2 =
    (pointsTo.newt(y_, w_) <<= (pointsTo.delta(z_, w_), pointsTo.full(x_, z_), load.full(y_, x_))) |
    plan(z_, x_, w_, y_);

// Rule 3: PointsTo(z, w) :- Store(y, x), PointsTo(y, z), PointsTo(x, w)
const auto store_variant1 = (pointsTo.newt(z_, w_) <<=
                             (pointsTo.delta(y_, z_), store.full(y_, x_), pointsTo.full(x_, w_))) |
                            plan(y_, x_, z_, w_);
const auto store_variant2 = (pointsTo.newt(z_, w_) <<=
                             (pointsTo.delta(x_, w_), store.full(y_, x_), pointsTo.full(y_, z_))) |
                            plan(x_, y_, w_, z_);

using AndersenBase = decltype(non_iterative(andersen_base_rule));
using AndersenFixpoint =
    decltype(fixpoint(assign_rule, load_variant1, load_variant2, store_variant1, store_variant2));
