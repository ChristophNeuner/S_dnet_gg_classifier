import sys 

import cytomine
from cytomine import Cytomine
from cytomine.models import CurrentUser, Project, ProjectCollection, ImageInstanceCollection, AnnotationCollection
from cytomine.models import Annotation, AnnotationTerm
from cytomine.models.ontology import Ontology, Term, RelationTerm, TermCollection, OntologyCollection
from cytomine import CytomineJob
from cytomine.utilities.descriptor_reader import read_descriptor

sys.path.append('../')
sys.path.append('../wsi_processing_pipeline/')
sys.path.append('../wsi_processing_pipeline/tile_extraction/')
import tile_extraction
from tile_extraction import tiles, util, slide
import wsi_processing_pipeline
from wsi_processing_pipeline import shared
from wsi_processing_pipeline.shared import roi
from wsi_processing_pipeline.cytomine import util_cytomine
from wsi_processing_pipeline.shared.util_shared import get_x, get_y, split


import fastai
from fastai import vision
from fastai.vision.all import *
from fastai.learner import *


import pathlib
from pathlib import Path
Path.ls = lambda x: [p for p in list(x.iterdir()) if '.ipynb_checkpoints' not in p.name]
from tqdm import tqdm


def main(argv):
     with CytomineJob.from_cli(argv) as cj:
        ##
        # get image instances
        ##
        print('get image instances')
        project_id = cj.parameters.cytomine_id_project
        project = Project().fetch(id=project_id)
        image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)
        
        ##
        # download wsi files #TODO map cytomine images dir into container
        ##
        print('download wsi files')
        wsi_tmp_dir = Path(f'./tmp_wsi')
        for i in image_instances:
            util_cytomine.download_wsi(image=i, directory=wsi_tmp_dir)
            
        wsi_paths = wsi_tmp_dir.ls()
        
        ##
        # get rois
        ##
        print('get rois')
        image_instance_to_rois = {}
        wsi_path_to_rois = {}
        for i in image_instances:
            wsi_path = util_cytomine.get_wsi_path_from_cytomine_image_instance(i=i, wsi_paths=wsi_paths)
            rois = util_cytomine.get_image_instance_annotations_as_rois(image=i, wsi_path=wsi_path)
            image_instance_to_rois[i] = rois
            wsi_path_to_rois[wsi_path] = rois
        
        ##    
        #tilesummaries
        ##
        print('tilesummaries')
        tilesummaries = tiles.WsisToTilesParallel(wsi_paths=wsi_paths,
                                          tile_height=1024, 
                                          tile_width=1024, 
                                          tile_scoring_function=tiles.score_tile_2,
                                          tile_score_thresh=0.0, 
                                          level=0, 
                                          wsi_path_to_rois=wsi_path_to_rois,
                                          minimal_tile_roi_intersection_ratio=0.9, 
                                          verbose=False, 
                                          grids_per_roi=1)
        
        ##
        # predictions
        ##
        print('predictions')
        exported_learner_path = Path('./dnet_vs_gg_resnet-1-resnet50_untrained.pkl')
        learner = load_learner(fname=exported_learner_path)
        
        for ts in tqdm(tilesummaries):  
            print(ts.wsi_path.name)        
            image = util_cytomine.get_cytomine_image_instances_for_wsi_name(wsi_name=ts.wsi_path.name, 
                                                                            projects=[project])[0]
            
            ###
            # delete all already existing tile annotations
            ###
            util_cytomine.delete_annotations(annotations=util_cytomine.get_annotations_with_term_filter(image=image, 
                                                         included_terms=[], 
                                                         excluded_terms=['roi']))
            
            for t in tqdm(ts.top_tiles()):
                if(t.predictions_fastai_inference is None):
                    t.predictions_fastai_inference = learner.predict(t)
                r = t.rectangle.as_roi(level=0, labels=list(t.predictions_fastai_inference[0]))         
                util_cytomine.add_rois_as_annotation_to_image(image=image, 
                                                rois=[r], 
                                                term_names=['tile'], 
                                                wsi_path=ts.wsi_path)
                
        ##
        # delete wsi (should be unnecessary, because files in docker containers are not persistent, 
        # if not stored in dockervolumes)
        ##
        print('delete wsi')
        for wp in wsi_paths:
            wp.delete()
        
        
      
if __name__ == "__main__":   
    main(sys.argv[1:])
