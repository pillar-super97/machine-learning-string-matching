# django imports
import csv, json, copy
from django.shortcuts import render
from django.views.generic import View
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from django.http.response import JsonResponse
from django.db import transaction

#project imports 
from .utils import  ngrams, tfidf_match
from .models import *


class UnSupervisedMatching(View):
    def post(self, request, *args, **kwargs):
        try:
            res_data , matched_obj_arr = {}, []
            request_data = request.POST.dict()
            if request_data.get("source") and request_data.get("target"):
                source_data = json.loads(request_data.get("source"))
                target_data = json.loads(request_data.get("target"))
            else:
                return JsonResponse(data='', message='The required fields are not provided', status=400)     
            source_data_list =  source_data.get('formatFields').copy()
            target_data_list = target_data.get('formatFields').copy()
            source_name = source_data.get("formatName")
            target_name = target_data.get("formatName")
            matched_df = tfidf_match(source_data_list, target_data_list)
            overall_confidence = matched_df["confidence"].sum()/len(matched_df)
            res_matched_data = matched_df.to_dict('records')
            with transaction.atomic():
                string_record = StringRecords.objects.create(source_format_name=source_name, 
                                            target_format_name=target_name, overall_confidence=overall_confidence)
                for data in res_matched_data:
                    if MatchedRecords.objects.filter(**data).exists():
                        matched_record = MatchedRecords.objects.get(**data)
                    else:
                        matched_record = MatchedRecords.objects.create(**data)
                    matched_obj_arr.append(matched_record)
                string_record.match_results.add(*matched_obj_arr)
            res_data.update(sourceFormatName=source_name, overall_confidence=overall_confidence,
                                targetformatName=target_name,mappings=res_matched_data)
            return JsonResponse(res_data, safe=False, status=200)
        except:
            return JsonResponse("Something went wrong", safe=False, status=400)


class AutomateMatching(View):
    def post(self, request, *args, **kwargs):
        try:
            res_data , matched_obj_arr = {}, []
            request_data = request.POST.dict()
            if request_data.get("source") and request_data.get("target"):
                source_data = json.loads(request_data.get("source"))
                target_data = json.loads(request_data.get("target"))
            else:
                return JsonResponse(data='', message='The required fields are not provided', status=400)     
            source_data_list =  source_data.get('formatFields').copy()
            target_data_list = target_data.get('formatFields').copy()
            source_name = source_data.get("formatName")
            target_name = target_data.get("formatName")
            matched_df = tfidf_match(source_data_list, target_data_list, flag="automate-match")
            res_matched_data = matched_df.to_dict('records')
            with transaction.atomic():
                for data in res_matched_data:
                    if TrainingService.objects.filter(**data).exists():
                        matched_record = TrainingService.objects.get(**data)
                    else:
                        matched_record = TrainingService.objects.create(**data)
            res_data.update(sourceFormatName=source_name, targetformatName=target_name, mappings=res_matched_data)
            return JsonResponse(res_data, safe=False, status=200)
        except:
            return JsonResponse("Something went wrong", safe=False, status=400)

        
        
      