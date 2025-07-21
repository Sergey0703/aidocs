# retrieval/results_fusion.py
# Advanced results fusion and ranking for multi-strategy retrieval
# ?????????? ????????: ?????? ????????? ??????? ??????????

import logging
import math
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class FusionResult:
    """Result of fusion process"""
    fused_results: List[Any]  # RetrievalResult objects
    fusion_method: str
    original_count: int
    final_count: int
    fusion_metadata: Dict[str, Any]
    fusion_time: float

class ResultsFusionEngine:
    """Advanced results fusion with multiple strategies"""
    
    def __init__(self, config):
        self.config = config
        
        # Fusion weights for different sources/methods
        self.method_weights = {
            "llamaindex_vector": 1.0,
            "llamaindex_vector_filtered": 1.2,
            "database_direct": 0.7,
            "database_direct_filtered": 0.9,
            "spacy": 0.8,
            "hybrid": 1.1
        }
        
        # Quality indicators for boosting
        self.quality_indicators = {
            "exact_match": 1.3,
            "high_confidence": 1.2,
            "content_filtered": 1.15,
            "known_entity": 1.25,
            "multiple_terms": 1.1
        }
    
    def fuse_results(self, 
                    all_results: List[Any], 
                    original_query: str,
                    extracted_entity: Optional[str] = None,
                    required_terms: List[str] = None) -> FusionResult:
        """Main fusion method that selects best strategy"""
        
        import time
        start_time = time.time()
        
        if not all_results:
            return FusionResult(
                fused_results=[],
                fusion_method="empty",
                original_count=0,
                final_count=0,
                fusion_metadata={"reason": "no_results"},
                fusion_time=time.time() - start_time
            )
        
        original_count = len(all_results)
        
        # Remove exact duplicates first
        deduplicated = self._remove_exact_duplicates(all_results)
        
        # ????????: ?????????? ????? ????????? ??? ???????????
        # fusion_method = self._select_fusion_strategy(deduplicated, original_query)
        fusion_method = "simple_weighted"  # ?????????? ??????? ?????????
        
        # Apply selected fusion method
        if fusion_method == "weighted_score":
            fused_results = self._weighted_score_fusion(
                deduplicated, original_query, extracted_entity, required_terms
            )
        elif fusion_method == "rank_fusion":
            fused_results = self._reciprocal_rank_fusion(deduplicated, original_query)
        elif fusion_method == "semantic_fusion":
            fused_results = self._semantic_clustering_fusion(deduplicated, original_query)
        elif fusion_method == "hybrid_fusion":
            fused_results = self._hybrid_fusion(
                deduplicated, original_query, extracted_entity, required_terms
            )
        elif fusion_method == "simple_weighted":
            # ?????: ?????????? weighted ?????????
            fused_results = self._simple_weighted_fusion(
                deduplicated, original_query, extracted_entity, required_terms
            )
        else:
            # Default: simple similarity sorting
            fused_results = sorted(deduplicated, key=lambda x: x.similarity_score, reverse=True)
        
        # Apply final filtering and limiting
        final_results = self._apply_final_filters(
            fused_results, original_query, extracted_entity, required_terms
        )
        
        fusion_time = time.time() - start_time
        
        logger.info(f"?? Fusion completed: {fusion_method} | {original_count}?{len(final_results)} results in {fusion_time:.3f}s")
        
        return FusionResult(
            fused_results=final_results,
            fusion_method=fusion_method,
            original_count=original_count,
            final_count=len(final_results),
            fusion_metadata=self._generate_fusion_metadata(
                all_results, final_results, fusion_method
            ),
            fusion_time=fusion_time
        )
    
    # ???????? ????????????????: ??????? ??????????? ?????????
    # def _select_fusion_strategy(self, results: List[Any], query: str) -> str:
    #     """Intelligently select fusion strategy based on data"""
    #     if len(results) <= 1:
    #         return "simple"
    #     
    #     # Analyze result characteristics
    #     methods_count = len(set(r.source_method for r in results))
    #     has_high_confidence = any(r.similarity_score > 0.8 for r in results)
    #     score_variance = self._calculate_score_variance(results)
    #     
    #     # Decision logic
    #     if methods_count >= 3 and score_variance > 0.1:
    #         return "hybrid_fusion"
    #     elif methods_count >= 2 and has_high_confidence:
    #         return "rank_fusion"
    #     elif len(results) >= 10 and score_variance < 0.05:
    #         return "semantic_fusion"
    #     else:
    #         return "weighted_score"
    
    def _simple_weighted_fusion(self, 
                               results: List[Any], 
                               query: str,
                               extracted_entity: Optional[str] = None,
                               required_terms: List[str] = None) -> List[Any]:
        """?????: ?????????? weighted fusion ??? ??????????? ??????????"""
        
        query_lower = query.lower()
        entity_lower = extracted_entity.lower() if extracted_entity else ""
        
        for result in results:
            # Base weight from method
            method_weight = self.method_weights.get(result.source_method, 1.0)
            
            # Content analysis
            content_lower = f"{result.content} {result.full_content} {result.filename}".lower()
            
            # Simple quality boost
            quality_multiplier = 1.0
            
            # Exact query match boost
            if query_lower in content_lower:
                quality_multiplier *= 1.3
            
            # Entity match boost
            if entity_lower and entity_lower in content_lower:
                quality_multiplier *= 1.2
            
            # High confidence boost
            if result.similarity_score > 0.6:  # ??????? ? 0.8
                quality_multiplier *= 1.1
            
            # Calculate final weighted score
            weighted_score = result.similarity_score * method_weight * quality_multiplier
            
            # Store in metadata
            result.metadata.update({
                "weighted_score": weighted_score,
                "method_weight": method_weight,
                "quality_multiplier": quality_multiplier,
                "fusion_method": "simple_weighted"
            })
        
        # Sort by weighted score
        return sorted(results, key=lambda x: x.metadata.get("weighted_score", x.similarity_score), reverse=True)
    
    def _weighted_score_fusion(self, 
                              results: List[Any], 
                              query: str,
                              extracted_entity: Optional[str] = None,
                              required_terms: List[str] = None) -> List[Any]:
        """Advanced weighted score fusion with multiple factors"""
        
        query_lower = query.lower()
        entity_lower = extracted_entity.lower() if extracted_entity else ""
        required_terms_lower = [term.lower() for term in (required_terms or [])]
        
        for result in results:
            # Base weight from method
            method_weight = self.method_weights.get(result.source_method, 1.0)
            
            # Content analysis
            content_lower = f"{result.content} {result.full_content} {result.filename}".lower()
            
            # Quality indicators
            quality_multiplier = 1.0
            
            # Exact query match boost
            if query_lower in content_lower:
                quality_multiplier *= self.quality_indicators["exact_match"]
            
            # Entity match boost
            if entity_lower and entity_lower in content_lower:
                quality_multiplier *= self.quality_indicators["known_entity"]
            
            # Required terms coverage
            if required_terms_lower:
                found_terms = sum(1 for term in required_terms_lower if term in content_lower)
                term_coverage = found_terms / len(required_terms_lower)
                if term_coverage > 0.5:
                    quality_multiplier *= (1.0 + term_coverage * 0.3)
            
            # High confidence boost
            if result.similarity_score > 0.8:
                quality_multiplier *= self.quality_indicators["high_confidence"]
            
            # Content filtering boost
            if result.metadata.get("content_filtered"):
                quality_multiplier *= self.quality_indicators["content_filtered"]
            
            # Length penalty for very short results
            content_length = len(result.full_content)
            if content_length < 50:
                quality_multiplier *= 0.8
            elif content_length > 1000:
                quality_multiplier *= 1.1
            
            # Calculate final weighted score
            weighted_score = result.similarity_score * method_weight * quality_multiplier
            
            # Store in metadata for debugging
            result.metadata.update({
                "weighted_score": weighted_score,
                "method_weight": method_weight,
                "quality_multiplier": quality_multiplier,
                "fusion_factors": {
                    "exact_match": query_lower in content_lower,
                    "entity_match": entity_lower in content_lower if entity_lower else False,
                    "term_coverage": found_terms / len(required_terms_lower) if required_terms_lower else 0,
                    "high_confidence": result.similarity_score > 0.8,
                    "content_filtered": result.metadata.get("content_filtered", False)
                }
            })
        
        # Sort by weighted score
        return sorted(results, key=lambda x: x.metadata.get("weighted_score", x.similarity_score), reverse=True)
    
    def _reciprocal_rank_fusion(self, results: List[Any], query: str) -> List[Any]:
        """Reciprocal Rank Fusion (RRF) for combining different ranking methods"""
        
        # Group results by method
        method_groups = defaultdict(list)
        for result in results:
            method_groups[result.source_method].append(result)
        
        # Sort each group independently  
        for method in method_groups:
            method_groups[method].sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Calculate RRF scores
        rrf_scores = {}
        k = 60  # RRF constant
        
        # Create unique identifier for each result
        for result in results:
            result_id = self._create_result_id(result)
            
            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    "result": result,
                    "rrf_score": 0,
                    "ranks": {},
                    "methods": set()
                }
            
            # Find rank in its method group
            method_list = method_groups[result.source_method]
            try:
                rank = next(i for i, r in enumerate(method_list) if self._create_result_id(r) == result_id) + 1
                rrf_contribution = 1.0 / (k + rank)
                
                rrf_scores[result_id]["rrf_score"] += rrf_contribution
                rrf_scores[result_id]["ranks"][result.source_method] = rank
                rrf_scores[result_id]["methods"].add(result.source_method)
                
            except (StopIteration, ValueError):
                logger.warning(f"Could not find rank for result in RRF")
        
        # Boost results that appear in multiple methods
        for result_id in rrf_scores:
            method_count = len(rrf_scores[result_id]["methods"])
            if method_count > 1:
                rrf_scores[result_id]["rrf_score"] *= (1.0 + (method_count - 1) * 0.2)
        
        # Sort by RRF score
        sorted_items = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Add RRF metadata to results
        fused_results = []
        for item in sorted_items:
            result = item["result"]
            result.metadata.update({
                "rrf_score": item["rrf_score"],
                "method_ranks": item["ranks"],
                "methods_count": len(item["methods"]),
                "fusion_method": "rrf"
            })
            fused_results.append(result)
        
        return fused_results
    
    def _semantic_clustering_fusion(self, results: List[Any], query: str) -> List[Any]:
        """Group semantically similar results and pick best from each cluster"""
        
        # ????????: ??????? semantic clustering ??? ??????????? - ?? ??????? ??????????
        logger.info("?? DIAGNOSTIC: Skipping semantic clustering - returning sorted results")
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        # ????????????????: ???????????? ?????? clustering
        # if len(results) < 5:
        #     return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        # 
        # # Simple clustering based on content similarity
        # clusters = []
        # used_indices = set()
        # 
        # for i, result in enumerate(results):
        #     if i in used_indices:
        #         continue
        #     
        #     # Start new cluster
        #     cluster = [result]
        #     used_indices.add(i)
        #     
        #     # Find similar results
        #     for j, other_result in enumerate(results[i+1:], i+1):
        #         if j in used_indices:
        #             continue
        #         
        #         if self._are_semantically_similar(result, other_result):
        #             cluster.append(other_result)
        #             used_indices.add(j)
        #     
        #     clusters.append(cluster)
        # 
        # # Pick best result from each cluster
        # cluster_representatives = []
        # for cluster in clusters:
        #     # Sort cluster by combined score
        #     cluster.sort(key=lambda x: (
        #         x.similarity_score * self.method_weights.get(x.source_method, 1.0)
        #     ), reverse=True)
        #     
        #     best_result = cluster[0]
        #     best_result.metadata.update({
        #         "cluster_size": len(cluster),
        #         "cluster_variants": len(set(r.filename for r in cluster)),
        #         "fusion_method": "semantic_clustering"
        #     })
        #     
        #     cluster_representatives.append(best_result)
        # 
        # # Sort clusters by their best representative's score
        # return sorted(cluster_representatives, key=lambda x: x.similarity_score, reverse=True)
    
    def _hybrid_fusion(self, 
                      results: List[Any], 
                      query: str,
                      extracted_entity: Optional[str] = None,
                      required_terms: List[str] = None) -> List[Any]:
        """Hybrid fusion combining multiple strategies"""
        
        # Apply weighted scoring
        weighted_results = self._weighted_score_fusion(
            results[:], query, extracted_entity, required_terms
        )
        
        # Apply RRF 
        rrf_results = self._reciprocal_rank_fusion(results[:], query)
        
        # Create hybrid score combining both approaches
        result_scores = {}
        
        for i, result in enumerate(weighted_results):
            result_id = self._create_result_id(result)
            weighted_rank = i + 1
            weighted_score = result.metadata.get("weighted_score", result.similarity_score)
            
            result_scores[result_id] = {
                "result": result,
                "weighted_rank": weighted_rank,
                "weighted_score": weighted_score,
                "rrf_score": 0,
                "rrf_rank": len(rrf_results) + 1
            }
        
        for i, result in enumerate(rrf_results):
            result_id = self._create_result_id(result)
            if result_id in result_scores:
                result_scores[result_id]["rrf_score"] = result.metadata.get("rrf_score", 0)
                result_scores[result_id]["rrf_rank"] = i + 1
        
        # Calculate hybrid score
        for result_id in result_scores:
            item = result_scores[result_id]
            
            # Normalize ranks (lower is better)
            weighted_rank_norm = 1.0 / item["weighted_rank"]
            rrf_rank_norm = 1.0 / item["rrf_rank"]
            
            # Combine with weights (favor weighted approach slightly)
            hybrid_score = (
                0.6 * weighted_rank_norm +
                0.4 * rrf_rank_norm +
                0.1 * item["weighted_score"] +
                0.1 * item["rrf_score"]
            )
            
            item["hybrid_score"] = hybrid_score
            item["result"].metadata.update({
                "hybrid_score": hybrid_score,
                "fusion_method": "hybrid",
                "weighted_rank": item["weighted_rank"],
                "rrf_rank": item["rrf_rank"]
            })
        
        # Sort by hybrid score
        sorted_items = sorted(
            result_scores.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        return [item["result"] for item in sorted_items]
    
    def _remove_exact_duplicates(self, results: List[Any]) -> List[Any]:
        """Remove exact duplicates based on content hash"""
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            # Create hash from content and filename
            content_hash = hash((result.full_content, result.filename))
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append(result)
            else:
                # Keep the one with better score/method
                existing_idx = None
                for i, existing in enumerate(unique_results):
                    if hash((existing.full_content, existing.filename)) == content_hash:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    existing = unique_results[existing_idx]
                    if (result.similarity_score > existing.similarity_score or
                        self._is_better_method(result.source_method, existing.source_method)):
                        unique_results[existing_idx] = result
        
        logger.info(f"?? Deduplication: {len(results)} ? {len(unique_results)} unique results")
        return unique_results
    
    def _apply_final_filters(self, 
                           results: List[Any], 
                           query: str,
                           extracted_entity: Optional[str] = None,
                           required_terms: List[str] = None) -> List[Any]:
        """??????????: ????????? ?????? ?????????? ??? ???????????"""
        
        if not results:
            return results
        
        # ??????????: ??????? ??????????? ?????????? ?? ????????  
        # min_score = max(0.1, results[0].similarity_score * 0.3)  # ??????: ??????? ??????
        min_score = 0.1  # ?????: ????? ?????? ?????
        filtered_results = [r for r in results if r.similarity_score >= min_score]
        
        # ??????????: ??????????? ????? ???????????
        # max_results = self.config.search.max_final_results  # ??????: ????? ???? ??????? ????  
        max_results = 15  # ?????: ?????????? ?????? ???????????
        final_results = filtered_results[:max_results]
        
        logger.info(f"?? Final filtering (RELAXED): {len(results)} ? {len(final_results)} results")
        logger.info(f"   Min score threshold: {min_score}, Max results: {max_results}")
        
        return final_results
    
    def _calculate_score_variance(self, results: List[Any]) -> float:
        """Calculate variance in similarity scores"""
        if len(results) <= 1:
            return 0.0
        
        scores = [r.similarity_score for r in results]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        return math.sqrt(variance)  # Return standard deviation
    
    def _create_result_id(self, result: Any) -> str:
        """Create unique identifier for result"""
        return f"{result.filename}_{hash(result.full_content[:100])}"
    
    def _are_semantically_similar(self, result1: Any, result2: Any, threshold: float = 0.7) -> bool:
        """Check if two results are semantically similar (simple implementation)"""
        # Simple similarity based on shared words
        words1 = set(result1.full_content.lower().split())
        words2 = set(result2.full_content.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity > threshold
    
    def _is_better_method(self, method1: str, method2: str) -> bool:
        """Compare which method is better"""
        return self.method_weights.get(method1, 1.0) > self.method_weights.get(method2, 1.0)
    
    def _generate_fusion_metadata(self, 
                                 original_results: List[Any],
                                 final_results: List[Any],
                                 fusion_method: str) -> Dict[str, Any]:
        """Generate metadata about fusion process"""
        
        original_methods = Counter(r.source_method for r in original_results)
        final_methods = Counter(r.source_method for r in final_results)
        
        if final_results:
            avg_score = sum(r.similarity_score for r in final_results) / len(final_results)
            score_range = (
                min(r.similarity_score for r in final_results),
                max(r.similarity_score for r in final_results)
            )
        else:
            avg_score = 0.0
            score_range = (0.0, 0.0)
        
        return {
            "fusion_method": fusion_method,
            "original_methods": dict(original_methods),
            "final_methods": dict(final_methods),
            "deduplication_ratio": len(final_results) / len(original_results) if original_results else 0,
            "avg_final_score": avg_score,
            "score_range": score_range,
            "quality_distribution": self._analyze_quality_distribution(final_results),
            "diagnostic_mode": True  # ?????????: ????????? ??? ? ??????????????? ??????
        }
    
    def _analyze_quality_distribution(self, results: List[Any]) -> Dict[str, int]:
        """Analyze quality distribution of final results"""
        if not results:
            return {"high": 0, "medium": 0, "low": 0}
        
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for result in results:
            score = result.similarity_score
            if score >= 0.7:
                distribution["high"] += 1
            elif score >= 0.4:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution