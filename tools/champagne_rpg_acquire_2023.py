#!/usr/bin/env python3
"""
Acquire the open same-year 2023 Champagne vineyard proxy source family:
- classic RPG 2023 via IGN API Carto
- RPG complété 2023 via Recherche Data Gouv / INRAE
This script acquires and qualifies source bytes. It does NOT claim CVI authority.
"""
from __future__ import annotations
import json, time, hashlib, zipfile
from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

OUT = Path("champagne_rpg_2023_output")
RAW = OUT/"raw"
META = OUT/"metadata"
DERIVED = OUT/"derived"
QA = OUT/"qa"
for p in [RAW, META, DERIVED, QA]: p.mkdir(parents=True, exist_ok=True)

SELECTOR_CODES = ['02042','02051','02053','02084','02094','02098','02114','02146','02163','02166','02168','02186','02187','02209','02213','02228','02239','02242','02268','02290','02292','02328','02347','02389','02484','02510','02521','02524','02540','02554','02555','02595','02596','02645','02653','02677','02701','02748','02818','10002','10007','10008','10011','10012','10022','10025','10029','10032','10033','10034','10039','10041','10048','10058','10068','10069','10070','10071','10076','10079','10097','10102','10103','10111','10113','10119','10126','10136','10137','10141','10150','10155','10160','10170','10176','10187','10197','10199','10232','10242','10248','10250','10261','10262','10264','10288','10295','10296','10306','10317','10330','10364','10366','10374','10384','10390','10404','10420','10427','10438','10439','10440','51002','51005','51007','51014','51020','51028','51029','51030','51036','51038','51039','51040','51042','51044','51045','51048','51049','51050','51052','51056','51058','51061','51069','51072','51073','51076','51079','51081','51085','51088','51089','51090','51092','51093','51102','51103','51105','51109','51111','51112','51119','51120','51121','51122','51124','51136','51140','51142','51145','51152','51153','51157','51158','51163','51171','51172','51173','51177','51181','51186','51188','51190','51192','51194','51195','51196','51198','51199','51200','51201','51202','51204','51210','51217','51218','51225','51230','51238','51239','51245','51247','51249','51252','51254','51256','51266','51267','51273','51275','51281','51282','51287','51291','51294','51298','51305','51308','51309','51310','51314','51320','51321','51325','51327','51328','51333','51338','51342','51344','51346','51348','51362','51363','51364','51365','51367','51374','51375','51376','51378','51379','51384','51387','51390','51392','51393','51396','51398','51403','51410','51413','51414','51416','51418','51421','51422','51425','51429','51431','51437','51440','51444','51445','51448','51450','51454','51457','51461','51465','51466','51468','51471','51472','51479','51480','51484','51496','51518','51523','51526','51527','51529','51532','51534','51535','51536','51558','51562','51563','51564','51568','51576','51577','51580','51581','51582','51584','51585','51586','51589','51590','51591','51592','51597','51599','51601','51602','51605','51609','51611','51612','51613','51614','51622','51624','51627','51629','51631','51633','51636','51639','51641','51643','51644','51645','51647','51657','52140','52426','77117','77331','77397']
EXPECTED_DEPT_COUNTS = {"02":39,"10":63,"51":207,"52":2,"77":3}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent":"ChampagneWorldTwin/1.0 public-research-acquisition"})

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def md5(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def fetch_current_communes():
    rows=[]
    for i,code in enumerate(SELECTOR_CODES,1):
        url=f"https://geo.api.gouv.fr/communes/{code}?fields=nom,code,departement,contour&format=geojson&geometry=contour"
        for attempt in range(5):
            try:
                r=SESSION.get(url, timeout=60); r.raise_for_status(); obj=r.json()
                if obj.get("type")=="Feature":
                    props=obj.get("properties",{}); geom=obj.get("geometry"); name=props.get("nom")
                    dep=(props.get("departement") or {}).get("code") if isinstance(props.get("departement"),dict) else None
                else:
                    name=obj.get("nom"); dep=(obj.get("departement") or {}).get("code") if isinstance(obj.get("departement"),dict) else None
                    geom=obj.get("contour") or obj.get("geometry")
                if not geom: raise ValueError(f"no geometry for {code}")
                rows.append({"geometry_code":code,"commune_name":name or code,"department_code":dep or code[:2],"geometry":shape(geom)})
                break
            except Exception:
                if attempt==4: raise
                time.sleep(2*(attempt+1))
    g=gpd.GeoDataFrame(rows,geometry="geometry",crs=4326)
    counts=g.groupby("department_code").size().to_dict()
    if counts != EXPECTED_DEPT_COUNTS: raise RuntimeError(f"selector department counts mismatch: {counts}")
    g.to_file(DERIVED/"selector_geoapi_current.gpkg",layer="selector",driver="GPKG")
    return g

def api_rpg_query(geom, code_cultu=None, limit=1000, start=0):
    payload={"annee":2023,"geom":mapping(geom),"_limit":limit,"_start":start}
    if code_cultu: payload["code_cultu"]=code_cultu
    r=SESSION.post("https://apicarto.ign.fr/api/rpg/v2",json=payload,timeout=180); r.raise_for_status(); return r

def probe_rpg(selector):
    av=selector[selector.geometry_code=="51029"]
    if av.empty: av=selector.iloc[[0]]
    r=api_rpg_query(av.iloc[0].geometry)
    (RAW/"classic_rpg_2023_probe.json").write_bytes(r.content)
    feats=r.json().get("features",[])
    props=[f.get("properties",{}) for f in feats]
    vc=[p for p in props if str(p.get("code_cultu","")).upper()=="VRC"]
    out={"feature_count":len(feats),"vrc_count":len(vc),"vrc_code_groups":sorted({str(p.get("code_group")) for p in vc if p.get("code_group") is not None})}
    (QA/"classic_rpg_2023_probe.json").write_text(json.dumps(out,indent=2))
    if not vc: raise RuntimeError("2023 probe did not expose VRC; fail closed")
    return out

def acquire_classic(selector):
    rawdir=RAW/"classic_rpg_2023_api_carto"; rawdir.mkdir(exist_ok=True)
    features=[]; requests_log=[]
    for _,row in selector.iterrows():
        start=0; page=0
        while True:
            for attempt in range(6):
                try:
                    r=api_rpg_query(row.geometry,"VRC",1000,start); break
                except Exception:
                    if attempt==5: raise
                    time.sleep(min(60,3*(2**attempt)))
            p=rawdir/f"{row.geometry_code}_{page:03d}.json"; p.write_bytes(r.content)
            feats=r.json().get("features",[])
            features.extend(feats)
            requests_log.append({"geometry_code":row.geometry_code,"page":page,"start":start,"n_features":len(feats),"sha256":sha256(p),"bytes":p.stat().st_size})
            if len(feats)<1000: break
            start+=1000; page+=1
    pd.DataFrame(requests_log).to_csv(META/"classic_rpg_2023_request_pages.csv",index=False)
    seen=set(); uniq=[]
    for f in features:
        props=f.get("properties",{})
        key=("feature_id",str(f["id"])) if f.get("id") is not None else None
        if key is None:
            for k in ["id_parcel","id_parcelle","id","ID_PARCEL","fid","uid"]:
                if props.get(k) not in (None,""): key=(k,str(props[k])); break
        if key is None: key=("hash",hashlib.sha256(json.dumps(f,sort_keys=True,separators=(",",":")).encode()).hexdigest())
        if key in seen: continue
        seen.add(key); uniq.append(f)
    if not uniq: raise RuntimeError("classic RPG 2023 VRC acquisition returned zero features")
    allpath=RAW/"classic_rpg_2023_vrc_dedup.geojson"
    allpath.write_text(json.dumps({"type":"FeatureCollection","features":uniq},separators=(",",":")))
    g=gpd.GeoDataFrame.from_features(uniq,crs=4326).to_crs(2154)
    g["source_family"]="CLASSIC_RPG_2023"; g["geometry_area_ha"]=g.geometry.area/10000
    g.to_file(DERIVED/"classic_rpg_2023_vineyard.gpkg",layer="classic_rpg_2023_vineyard",driver="GPKG")
    return g

def dataverse_meta(pid):
    r=SESSION.get("https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/",params={"persistentId":pid},timeout=120)
    r.raise_for_status(); return r.json()

def find_file(meta,predicate):
    files=meta["data"]["latestVersion"]["files"]; matches=[]
    for x in files:
        df=x.get("dataFile",{}); label=x.get("label") or df.get("filename") or ""
        if predicate(label,df): matches.append((x,df,label))
    if len(matches)!=1: raise RuntimeError(f"file match failure: matches={[m[2] for m in matches]}, all={[(x.get('label') or x.get('dataFile',{}).get('filename')) for x in files]}")
    return matches[0]

def download_dataverse(pid,pred,name,expected_md5=None):
    meta=dataverse_meta(pid); (META/(name+".dataset.json")).write_text(json.dumps(meta,indent=2))
    _,df,label=find_file(meta,pred); fid=df["id"]; out=RAW/name
    with SESSION.get(f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{fid}",stream=True,timeout=300) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
    got=md5(out); provider=df.get("md5")
    if expected_md5 and got.lower()!=expected_md5.lower(): raise RuntimeError(f"MD5 mismatch {name}")
    if provider and got.lower()!=str(provider).lower(): raise RuntimeError(f"provider MD5 mismatch {name}")
    return out,{"pid":pid,"file_id":fid,"label":label,"bytes":out.stat().st_size,"md5":got,"sha256":sha256(out),"provider_md5":provider}

def extract_zip(z,key):
    d=RAW/(key+"_extracted"); d.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as zz: zz.extractall(d)
    return d

def load_completed_vines(folder,source_family):
    parts=[]; schema=[]
    for shp in folder.rglob("*.shp"):
        g=gpd.read_file(shp); schema.append({"file":str(shp),"n":len(g),"crs":str(g.crs),"columns":list(g.columns)})
        cols={c.lower():c for c in g.columns}
        cult=next((cols[k] for k in ["code_cultu","code_culture","culture","cultu"] if k in cols),None)
        group=next((cols[k] for k in ["code_group","code_groupe","groupe","group"] if k in cols),None)
        if cult: mask=g[cult].astype(str).str.upper().eq("VRC")
        elif group: mask=g[group].astype(str).isin(["21","21.0"])
        else: continue
        h=g[mask].copy()
        if len(h): h["source_family"]=source_family; parts.append(h)
    (META/(source_family+"_schema_inventory.json")).write_text(json.dumps(schema,indent=2))
    if not parts: raise RuntimeError(f"no vineyard rows identified for {source_family}")
    g=gpd.GeoDataFrame(pd.concat(parts,ignore_index=True),geometry="geometry",crs=parts[0].crs).to_crs(2154)
    g["geometry_area_ha"]=g.geometry.area/10000
    return g

def main():
    selector=fetch_current_communes(); probe_rpg(selector); classic=acquire_classic(selector)
    hdfz,hdfm=download_dataverse("doi:10.57745/VGIHYD",lambda l,d:l=="rpg_complete_2023_Region_32_Hauts-de-France.zip","rpg_complete_2023_Region_32_Hauts-de-France.zip","094a05fba0e1ec5d6a0d51f0b983ee81")
    idfz,idfm=download_dataverse("doi:10.57745/QLNGFX",lambda l,d:l=="rpg_complete_2023_Region_11_Ile-de-France.zip","rpg_complete_2023_Region_11_Ile-de-France.zip","003876d48b79089aa346f6a8ffc216cb")
    gez,gem=download_dataverse("doi:10.57745/A8TDAO",lambda l,d:l.lower().endswith(".zip") and ("partie_2" in l.lower() or "part_2" in l.lower() or "2_sur_2" in l.lower()),"rpg_complete_2023_Grand_Est_part2.zip")
    (META/"completed_rpg_download_receipt.json").write_text(json.dumps([hdfm,idfm,gem],indent=2))
    hdf=load_completed_vines(extract_zip(hdfz,"hdf"),"RPG_COMPLETE_2023_HDF")
    idf=load_completed_vines(extract_zip(idfz,"idf"),"RPG_COMPLETE_2023_IDF")
    ge=load_completed_vines(extract_zip(gez,"grand_est_p2"),"RPG_COMPLETE_2023_GRAND_EST_P2")
    completed=gpd.GeoDataFrame(pd.concat([hdf,idf,ge],ignore_index=True),geometry="geometry",crs=2154)
    completed.to_file(DERIVED/"rpg_complete_2023_vineyard.gpkg",layer="rpg_complete_2023_vineyard",driver="GPKG")

    classic_union=unary_union(classic.geometry.values)
    before=float(completed.geometry.area.sum()/10000)
    completed["geometry"]=completed.geometry.apply(lambda x:x.difference(classic_union))
    completed=completed[~completed.geometry.is_empty].copy()
    after=float(completed.geometry.area.sum()/10000)
    (QA/"overlap_audit.json").write_text(json.dumps({"completed_area_before_ha":before,"completed_area_after_ha":after,"overlap_removed_ha":before-after},indent=2))

    union=gpd.GeoDataFrame(pd.concat([classic,completed],ignore_index=True),geometry="geometry",crs=2154)
    union.to_file(DERIVED/"champagne_open_vineyard_proxy_2023_unclipped.gpkg",layer="vineyard_proxy",driver="GPKG")
    sel=selector.to_crs(2154)
    clipped=gpd.overlay(union,sel[["geometry_code","commune_name","department_code","geometry"]],how="intersection",keep_geom_type=False)
    clipped=clipped[~clipped.geometry.is_empty].copy(); clipped["area_ha"]=clipped.geometry.area/10000
    clipped.to_file(DERIVED/"champagne_open_vineyard_proxy_2023_geoapi_selector.gpkg",layer="vineyard_proxy_2023",driver="GPKG")
    comm=clipped.groupby(["geometry_code","commune_name","department_code","source_family"],as_index=False)["area_ha"].sum()
    comm.to_csv(DERIVED/"vineyard_area_by_current_commune_and_source.csv",index=False)
    clipped.groupby(["department_code","source_family"],as_index=False)["area_ha"].sum().to_csv(DERIVED/"vineyard_area_by_department_and_source.csv",index=False)
    clipped.groupby("source_family",as_index=False)["area_ha"].sum().to_csv(DERIVED/"source_family_contributions.csv",index=False)
    totals=comm.groupby(["geometry_code","commune_name","department_code"],as_index=False)["area_ha"].sum()
    selector[~selector.geometry_code.isin(set(totals.geometry_code))][["geometry_code","commune_name","department_code"]].to_csv(DERIVED/"zero_area_communes.csv",index=False)
    q={"status":"PASS_OPEN_PHYSICAL_VINEYARD_PROXY_2023_QUALIFIED_REMOTE_ACQUISITION","clock":2023,"selector_codes":len(selector),"classic_features":len(classic),"classic_area_ha":float(classic.geometry.area.sum()/10000),"completed_features_after_overlap_trim":len(completed),"completed_area_ha_after_overlap_trim":float(completed.geometry.area.sum()/10000),"overlap_removed_ha":before-after,"remote_geoapi_selector_intersection_area_ha":float(clipped.area_ha.sum()),"communes_with_proxy_area":int(totals.geometry_code.nunique()),"claim_cap":"OPEN_2023_VINEYARD_PROXY_ONLY_NOT_CVI_TRUTH_NOT_LEGAL_AOC_ELIGIBILITY_NOT_A7_PASS"}
    (QA/"qualification.json").write_text(json.dumps(q,indent=2))
    rows=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file(): rows.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    pd.DataFrame(rows).to_csv(OUT/"SHA256_MANIFEST.csv",index=False)
    print(json.dumps(q,indent=2))

if __name__=="__main__": main()
