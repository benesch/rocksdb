//  Copyright (c) 2016-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/range_del_aggregator.h"

#include <algorithm>

namespace rocksdb {

RangeDelAggregator::RangeDelAggregator(
    const InternalKeyComparator& icmp,
    const std::vector<SequenceNumber>& snapshots,
    bool collapse_deletions /* = true */)
    : upper_bound_(kMaxSequenceNumber),
      icmp_(icmp),
      collapse_deletions_(collapse_deletions) {
  InitRep(snapshots);
}

RangeDelAggregator::RangeDelAggregator(const InternalKeyComparator& icmp,
                                       SequenceNumber snapshot,
                                       bool collapse_deletions /* = false */)
    : upper_bound_(snapshot),
      icmp_(icmp),
      collapse_deletions_(collapse_deletions) {}

void RangeDelAggregator::InitRep(const std::vector<SequenceNumber>& snapshots) {
  assert(rep_ == nullptr);
  rep_.reset(new Rep());
  for (auto snapshot : snapshots) {
    rep_->stripe_map_.emplace(
        snapshot,
        TombstoneStripe(icmp_.user_comparator(), collapse_deletions_));
  }
  // Data newer than any snapshot falls in this catch-all stripe
  rep_->stripe_map_.emplace(
      kMaxSequenceNumber,
      TombstoneStripe(icmp_.user_comparator(), collapse_deletions_));
  rep_->pinned_iters_mgr_.StartPinning();
}

bool RangeDelAggregator::ShouldDeleteImpl(
    const Slice& internal_key, RangeDelAggregator::RangePositioningMode mode) {
  assert(rep_ != nullptr);
  ParsedInternalKey parsed;
  if (!ParseInternalKey(internal_key, &parsed)) {
    assert(false);
  }
  return ShouldDelete(parsed, mode);
}

bool RangeDelAggregator::ShouldDeleteImpl(
    const ParsedInternalKey& parsed,
    RangeDelAggregator::RangePositioningMode mode) {
  assert(IsValueType(parsed.type));
  assert(rep_ != nullptr);
  auto& positional_tombstone_map = GetPositionalTombstoneMap(parsed.sequence);
  if (positional_tombstone_map.Empty()) {
    return false;
  }
  if (!positional_tombstone_map.Valid() &&
      (mode == kForwardTraversal || mode == kBackwardTraversal)) {
    // invalid (e.g., if AddTombstones() changed the deletions), so need to
    // reseek
    mode = kBinarySearch;
  }
  switch (mode) {
    case kFullScan:
      assert(!collapse_deletions_);
      // The maintained state in PositionalTombstoneMap isn't useful when
      // we linear scan from the beginning each time, but we maintain it anyways
      // for consistency.
      positional_tombstone_map.SeekToFirst();
      while (positional_tombstone_map.Valid()) {
        const auto& tombstone = positional_tombstone_map.Tombstone();
        if (icmp_.user_comparator()->Compare(parsed.user_key,
                                             tombstone.start_key_) < 0) {
          break;
        }
        if (parsed.sequence < tombstone.seq_ &&
            icmp_.user_comparator()->Compare(parsed.user_key,
                                             tombstone.end_key_) < 0) {
          return true;
        }
        positional_tombstone_map.Next();
      }
      return false;
    case kForwardTraversal:
      assert(collapse_deletions_ && positional_tombstone_map.Valid());
      while (positional_tombstone_map.Valid() &&
             icmp_.user_comparator()->Compare(
               positional_tombstone_map.Tombstone().end_key_,
               parsed.user_key) <= 0) {
        positional_tombstone_map.Next();
      }
      break;
    case kBackwardTraversal:
      assert(collapse_deletions_ && positional_tombstone_map.Valid());
      while (positional_tombstone_map.Valid() &&
             icmp_.user_comparator()->Compare(parsed.user_key,
                                              positional_tombstone_map.Tombstone().start_key_) < 0) {
        positional_tombstone_map.Prev();
      }
      break;
    case kBinarySearch:
      assert(collapse_deletions_);
      positional_tombstone_map.Seek(parsed.user_key);
      break;
  }
  assert(mode != kFullScan);
  if (!positional_tombstone_map.Valid()) {
    // Seeked off the end. No tombstone applies.
    return false;
  }
  // assert(positional_tombstone_map.Valid() &&
  //        icmp_.user_comparator()->Compare(positional_tombstone_map.Tombstone().start_key_,
  //                                         parsed.user_key) <= 0);
  // assert(std::next(tombstone_map_iter) == tombstone_map.end() ||
  //        icmp_.user_comparator()->Compare(
  //            parsed.user_key, std::next(tombstone_map_iter)->first) < 0);
  return parsed.sequence < positional_tombstone_map.Tombstone().seq_;
}

bool RangeDelAggregator::IsRangeOverlapped(const Slice& start,
                                           const Slice& end) {
  // so far only implemented for non-collapsed mode since file ingestion (only
  //  client) doesn't use collapsing
  assert(!collapse_deletions_);
  if (rep_ == nullptr) {
    return false;
  }
  for (const auto& seqnum_and_tombstone_map : rep_->stripe_map_) {
    for (const auto& start_key_and_tombstone :
         seqnum_and_tombstone_map.second.raw_map) {
      const auto& tombstone = start_key_and_tombstone.second;
      if (icmp_.user_comparator()->Compare(start, tombstone.end_key_) < 0 &&
          icmp_.user_comparator()->Compare(tombstone.start_key_, end) <= 0 &&
          icmp_.user_comparator()->Compare(tombstone.start_key_,
                                           tombstone.end_key_) < 0) {
        return true;
      }
    }
  }
  return false;
}

bool RangeDelAggregator::ShouldAddTombstones(
    bool bottommost_level /* = false */) {
  // TODO(andrewkr): can we just open a file and throw it away if it ends up
  // empty after AddToBuilder()? This function doesn't take into subcompaction
  // boundaries so isn't completely accurate.
  if (rep_ == nullptr) {
    return false;
  }
  auto stripe_map_iter = rep_->stripe_map_.begin();
  assert(stripe_map_iter != rep_->stripe_map_.end());
  if (bottommost_level) {
    // For the bottommost level, keys covered by tombstones in the first
    // (oldest) stripe have been compacted away, so the tombstones are obsolete.
    ++stripe_map_iter;
  }
  while (stripe_map_iter != rep_->stripe_map_.end()) {
    if (!stripe_map_iter->second.Empty()) {
      return true;
    }
    ++stripe_map_iter;
  }
  return false;
}

Status RangeDelAggregator::AddTombstones(
    std::unique_ptr<InternalIterator> input) {
  if (input == nullptr) {
    return Status::OK();
  }
  input->SeekToFirst();
  bool first_iter = true;
  while (input->Valid()) {
    // The tombstone map holds slices into the iterator's memory. This assert
    // ensures pinning the iterator also pins the keys/values.
    assert(input->IsKeyPinned() && input->IsValuePinned());

    if (first_iter) {
      if (rep_ == nullptr) {
        InitRep({upper_bound_});
      } else {
        InvalidateTombstoneMapPositions();
      }
      first_iter = false;
    }
    ParsedInternalKey parsed_key;
    if (!ParseInternalKey(input->key(), &parsed_key)) {
      return Status::Corruption("Unable to parse range tombstone InternalKey");
    }
    RangeTombstone tombstone(parsed_key, input->value());
    GetStripe(tombstone.seq_).AddTombstone(std::move(tombstone));
    input->Next();
  }
  if (!first_iter) {
    rep_->pinned_iters_mgr_.PinIterator(input.release(), false /* arena */);
  }
  return Status::OK();
}

TombstoneStripe& RangeDelAggregator::GetStripe(SequenceNumber seq) {
  assert(rep_ != nullptr);
  // The stripe includes seqnum for the snapshot above and excludes seqnum for
  // the snapshot below.
  StripeMap::iterator iter;
  if (seq > 0) {
    // upper_bound() checks strict inequality so need to subtract one
    iter = rep_->stripe_map_.upper_bound(seq - 1);
  } else {
    iter = rep_->stripe_map_.begin();
  }
  // catch-all stripe justifies this assertion in either of above cases
  assert(iter != rep_->stripe_map_.end());
  return iter->second;
}

// // TODO(andrewkr): We should implement an iterator over range tombstones in our
// // map. It'd enable compaction to open tables on-demand, i.e., only once range
// // tombstones are known to be available, without the code duplication we have
// // in ShouldAddTombstones(). It'll also allow us to move the table-modifying
// // code into more coherent places: CompactionJob and BuildTable().
// void RangeDelAggregator::AddToBuilder(
//     TableBuilder* builder, const Slice* lower_bound, const Slice* upper_bound,
//     FileMetaData* meta,
//     CompactionIterationStats* range_del_out_stats /* = nullptr */,
//     bool bottommost_level /* = false */) {
//   if (rep_ == nullptr) {
//     return;
//   }
//   auto stripe_map_iter = rep_->stripe_map_.begin();
//   assert(stripe_map_iter != rep_->stripe_map_.end());
//   if (bottommost_level) {
//     // TODO(andrewkr): these are counted for each compaction output file, so
//     // lots of double-counting.
//     if (!stripe_map_iter->second.Empty()) {
//       range_del_out_stats->num_range_del_drop_obsolete +=
//           static_cast<int64_t>(stripe_map_iter->second.Size()) -
//           (collapse_deletions_ ? 1 : 0);
//       range_del_out_stats->num_record_drop_obsolete +=
//           static_cast<int64_t>(stripe_map_iter->second.Size()) -
//           (collapse_deletions_ ? 1 : 0);
//     }
//     // For the bottommost level, keys covered by tombstones in the first
//     // (oldest) stripe have been compacted away, so the tombstones are obsolete.
//     ++stripe_map_iter;
//   }

//   // Note the order in which tombstones are stored is insignificant since we
//   // insert them into a std::map on the read path.
//   while (stripe_map_iter != rep_->stripe_map_.end()) {
//     bool first_added = false;
//     auto& tombstone_map = stripe_map_iter->second;
//     for (tombstone_map.SeekToFirst(); tombstone_map.Valid(); tombstone_map.Next()) {
//       auto tombstone = tombstone_map.Tombstone();
//       if (upper_bound != nullptr &&
//           icmp_.user_comparator()->Compare(*upper_bound,
//                                            tombstone.start_key_) <= 0) {
//         // Tombstones starting at upper_bound or later only need to be included
//         // in the next table. Break because subsequent tombstones will start
//         // even later.
//         break;
//       }
//       if (lower_bound != nullptr &&
//           icmp_.user_comparator()->Compare(tombstone.end_key_,
//                                            *lower_bound) <= 0) {
//         // Tombstones ending before or at lower_bound only need to be included
//         // in the prev table. Continue because subsequent tombstones may still
//         // overlap [lower_bound, upper_bound).
//         continue;
//       }

//       auto ikey_and_end_key = tombstone.Serialize();
//       builder->Add(ikey_and_end_key.first.Encode(), ikey_and_end_key.second);
//       if (!first_added) {
//         first_added = true;
//         InternalKey smallest_candidate = std::move(ikey_and_end_key.first);
//         if (lower_bound != nullptr &&
//             icmp_.user_comparator()->Compare(smallest_candidate.user_key(),
//                                              *lower_bound) <= 0) {
//           // Pretend the smallest key has the same user key as lower_bound
//           // (the max key in the previous table or subcompaction) in order for
//           // files to appear key-space partitioned.
//           //
//           // Choose lowest seqnum so this file's smallest internal key comes
//           // after the previous file's/subcompaction's largest. The fake seqnum
//           // is OK because the read path's file-picking code only considers user
//           // key.
//           smallest_candidate = InternalKey(*lower_bound, 0, kTypeRangeDeletion);
//         }
//         if (meta->smallest.size() == 0 ||
//             icmp_.Compare(smallest_candidate, meta->smallest) < 0) {
//           meta->smallest = std::move(smallest_candidate);
//         }
//       }
//       InternalKey largest_candidate = tombstone.SerializeEndKey();
//       if (upper_bound != nullptr &&
//           icmp_.user_comparator()->Compare(*upper_bound,
//                                            largest_candidate.user_key()) <= 0) {
//         // Pretend the largest key has the same user key as upper_bound (the
//         // min key in the following table or subcompaction) in order for files
//         // to appear key-space partitioned.
//         //
//         // Choose highest seqnum so this file's largest internal key comes
//         // before the next file's/subcompaction's smallest. The fake seqnum is
//         // OK because the read path's file-picking code only considers the user
//         // key portion.
//         //
//         // Note Seek() also creates InternalKey with (user_key,
//         // kMaxSequenceNumber), but with kTypeDeletion (0x7) instead of
//         // kTypeRangeDeletion (0xF), so the range tombstone comes before the
//         // Seek() key in InternalKey's ordering. So Seek() will look in the
//         // next file for the user key.
//         largest_candidate = InternalKey(*upper_bound, kMaxSequenceNumber,
//                                         kTypeRangeDeletion);
//       }
//       if (meta->largest.size() == 0 ||
//           icmp_.Compare(meta->largest, largest_candidate) < 0) {
//         meta->largest = std::move(largest_candidate);
//       }
//       meta->smallest_seqno = std::min(meta->smallest_seqno, tombstone.seq_);
//       meta->largest_seqno = std::max(meta->largest_seqno, tombstone.seq_);
//     }
//     ++stripe_map_iter;
//   }
// }

// bool RangeDelAggregator::IsEmpty() {
//   if (rep_ == nullptr) {
//     return true;
//   }
//   for (auto stripe_map_iter = rep_->stripe_map_.begin();
//        stripe_map_iter != rep_->stripe_map_.end(); ++stripe_map_iter) {
//     if (!stripe_map_iter->second.Empty()) {
//       return false;
//     }
//   }
//   return true;
// }

// bool RangeDelAggregator::AddFile(uint64_t file_number) {
//   if (rep_ == nullptr) {
//     return true;
//   }
//   return rep_->added_files_.emplace(file_number).second;
// }

// RangeDelAggregator::PositionalTombstoneMap::PositionalTombstoneMap(
//     const Comparator* user_comparator, bool collapse_deletions)
//     : raw_map_(TombstoneMap()),
//       collapse_deletions_(collapse_deletions) {

// }

// bool RangeDelAggregator::PositionalTombstoneMap::Empty() const {
//   return raw_map_.empty();
// }

// size_t RangeDelAggregator::PositionalTombstoneMap::Size() const {
//   return raw_map_.size();
// }

// bool RangeDelAggregator::PositionalTombstoneMap::Valid() const {
//   return iter_ != raw_map_.end();
// }

// void RangeDelAggregator::PositionalTombstoneMap::Invalidate() {
//   iter_ = raw_map_.end();
// }

// void RangeDelAggregator::PositionalTombstoneMap::SeekToFirst() {
//   iter_ = raw_map_.begin();
// }

// void RangeDelAggregator::PositionalTombstoneMap::Seek(const Slice& target) {
//   assert(collapse_deletions_);
//   iter_ = raw_map_.upper_bound(target);
//   if (iter_ == raw_map_.begin()) {
//     Invalidate();
//   }
//   iter_--;
// }


TombstoneStripe::TombstoneStripe(const Comparator* user_comparator,
                                 bool collapsed)
    : ucmp_(user_comparator),
      collapsed_(collapsed),
      tombstones_(stl_wrappers::LessOfComparator(user_comparator)) {}

void TombstoneStripe::AddTombstone(RangeTombstone tombstone) {
  if (!collapsed_) {
    tombstones_.emplace(tombstone.start_key_, std::move(tombstone));
    return;
  }

  // In collapsed mode, we only fill the seq_ field in the TombstoneMap's
  // values. The end_key is unneeded because we assume the tombstone extends
  // until the next tombstone starts. For gaps between real tombstones and for
  // the last real tombstone, we denote end keys by inserting fake tombstones
  // with sequence number zero.
  std::vector<RangeTombstone> new_range_dels{
      tombstone, RangeTombstone(tombstone.end_key_, Slice(), 0)};
  auto new_range_dels_iter = new_range_dels.begin();
  // Position at the first overlapping existing tombstone; if none exists,
  // insert until we find an existing one overlapping a new point
  const Slice* tombstones_begin = nullptr;
  if (!tombstones_.empty()) {
    tombstones_begin = &tombstones_.begin()->first;
  }
  auto last_range_dels_iter = new_range_dels_iter;
  while (new_range_dels_iter != new_range_dels.end() &&
          (tombstones_begin == nullptr ||
          ucmp_->Compare(new_range_dels_iter->start_key_,
                         *tombstones_begin) < 0)) {
    tombstones_.emplace(
        new_range_dels_iter->start_key_,
        RangeTombstone(Slice(), Slice(), new_range_dels_iter->seq_));
    last_range_dels_iter = new_range_dels_iter;
    ++new_range_dels_iter;
  }
  if (new_range_dels_iter == new_range_dels.end()) {
    return;
  }
  // above loop advances one too far
  new_range_dels_iter = last_range_dels_iter;
  auto tombstones_iter =
      tombstones_.upper_bound(new_range_dels_iter->start_key_);
  // if nothing overlapped we would've already inserted all the new points
  // and returned early
  assert(tombstones_iter != tombstones_.begin());
  tombstones_iter--;

  // untermed_seq is non-kMaxSequenceNumber when we covered an existing point
  // but haven't seen its corresponding endpoint. It's used for (1) deciding
  // whether to forcibly insert the new interval's endpoint; and (2) possibly
  // raising the seqnum for the to-be-inserted element (we insert the max seqnum
  // between the next new interval and the unterminated interval).
  SequenceNumber untermed_seq = kMaxSequenceNumber;
  while (tombstones_iter != tombstones_.end() &&
         new_range_dels_iter != new_range_dels.end()) {
    const Slice *tombstones_iter_end = nullptr,
                *new_range_dels_iter_end = nullptr;
    if (tombstones_iter != tombstones_.end()) {
      auto next_tombstones_iter = std::next(tombstones_iter);
      if (next_tombstones_iter != tombstones_.end()) {
        tombstones_iter_end = &next_tombstones_iter->first;
      }
    }
    if (new_range_dels_iter != new_range_dels.end()) {
      auto next_new_range_dels_iter = std::next(new_range_dels_iter);
      if (next_new_range_dels_iter != new_range_dels.end()) {
        new_range_dels_iter_end = &next_new_range_dels_iter->start_key_;
      }
    }

    // our positions in existing/new tombstone collections should always
    // overlap. The non-overlapping cases are handled above and below this loop.
    assert(new_range_dels_iter_end == nullptr ||
            ucmp_->Compare(tombstones_iter->first,
                           *new_range_dels_iter_end) < 0);
    assert(tombstone_map_iter_end == nullptr ||
            ucmp_->Compare(new_range_dels_iter->start_key_,
                           *tombstones_iter_end) < 0);

    int new_to_old_start_cmp = ucmp_->Compare(
        new_range_dels_iter->start_key_, tombstones_iter->first);
    // nullptr end means extends infinitely rightwards, set new_to_old_end_cmp
    // accordingly so we can use common code paths later.
    int new_to_old_end_cmp;
    if (new_range_dels_iter_end == nullptr && tombstones_iter_end == nullptr) {
      new_to_old_end_cmp = 0;
    } else if (new_range_dels_iter_end == nullptr) {
      new_to_old_end_cmp = 1;
    } else if (tombstones_iter_end == nullptr) {
      new_to_old_end_cmp = -1;
    } else {
      new_to_old_end_cmp = ucmp_->Compare(
          *new_range_dels_iter_end, *tombstones_iter_end);
    }

    if (new_to_old_start_cmp < 0) {
      // the existing one's left endpoint comes after, so raise/delete it if
      // it's covered.
      if (tombstones_iter->second.seq_ < new_range_dels_iter->seq_) {
        untermed_seq = tombstones_iter->second.seq_;
        if (tombstones_iter != tombstones_.begin() &&
            std::prev(tombstones_iter)->second.seq_ ==
                new_range_dels_iter->seq_) {
          tombstones_iter = tombstones_.erase(tombstones_iter);
          --tombstones_iter;
        } else {
          tombstones_iter->second.seq_ = new_range_dels_iter->seq_;
        }
      }
    } else if (new_to_old_start_cmp > 0) {
      if (untermed_seq != kMaxSequenceNumber ||
          tombstones_iter->second.seq_ < new_range_dels_iter->seq_) {
        auto seq = tombstones_iter->second.seq_;
        // need to adjust this element if not intended to span beyond the new
        // element (i.e., was_tombstone_map_iter_raised == true), or if it can
        // be raised
        tombstones_iter = tombstones_.emplace(
            new_range_dels_iter->start_key_,
            RangeTombstone(
                Slice(), Slice(),
                std::max(
                    untermed_seq == kMaxSequenceNumber ? 0 : untermed_seq,
                    new_range_dels_iter->seq_)));
        untermed_seq = seq;
      }
    } else {
      // their left endpoints coincide, so raise the existing one if needed
      if (tombstones_iter->second.seq_ < new_range_dels_iter->seq_) {
        untermed_seq = tombstones_iter->second.seq_;
        tombstones_iter->second.seq_ = new_range_dels_iter->seq_;
      }
    }

    // advance whichever one ends earlier, or both if their right endpoints
    // coincide
    if (new_to_old_end_cmp < 0) {
      ++new_range_dels_iter;
    } else if (new_to_old_end_cmp > 0) {
      ++tombstones_iter;
      untermed_seq = kMaxSequenceNumber;
    } else {
      ++new_range_dels_iter;
      ++tombstones_iter;
      untermed_seq = kMaxSequenceNumber;
    }
  }
  while (new_range_dels_iter != new_range_dels.end()) {
    tombstones_.emplace(
        new_range_dels_iter->start_key_,
        RangeTombstone(Slice(), Slice(), new_range_dels_iter->seq_));
    ++new_range_dels_iter;
  }
}

}  // namespace rocksdb
