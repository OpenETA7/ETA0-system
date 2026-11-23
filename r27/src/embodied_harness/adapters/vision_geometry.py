"""Unlabelled visible RGB-D components. No simulator segmentation or object metadata."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def world_cloud(depth,fovy,rotation,position):
    h,w=depth.shape;v,u=np.mgrid[:h,:w];f=.5*h/np.tan(np.deg2rad(fovy)/2)
    camera=np.stack(((u-w/2)/f*depth,-(v-h/2)/f*depth,-depth),axis=-1)
    return camera@np.asarray(rotation).T+np.asarray(position)


def propose_surfaces(depth,fovy,rotation,position,support,robot_position):
    if support is None:return [],None,None
    cloud=world_cloud(depth,fovy,rotation,position)
    elevated=(cloud[...,2]>support+.005)&(cloud[...,2]<support+.35)
    elevated &= np.linalg.norm(cloud[...,:2]-np.asarray(robot_position)[:2],axis=-1)<.8
    elevated &= np.isfinite(cloud).all(axis=-1)
    # Image overlap must not merge distinct objects at different depths (e.g.
    # a can behind a carton). Connect neighbours only when their 3-D points agree.
    mask=elevated
    index=np.full(mask.shape,-1,dtype=int);index[mask]=np.arange(mask.sum())
    rows=[];cols=[]
    for a,b,pa,pb in [(index[:,:-1],index[:,1:],cloud[:,:-1],cloud[:,1:]),
                      (index[:-1,:],index[1:,:],cloud[:-1,:],cloud[1:,:])]:
        valid=(a>=0)&(b>=0)&(np.linalg.norm(pa-pb,axis=-1)<.012)
        rows.extend(a[valid]);cols.extend(b[valid])
    graph=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(int(mask.sum()),int(mask.sum())))
    count,components=connected_components(graph,directed=False)
    labels=np.zeros(mask.shape,dtype=int);labels[mask]=components+1
    candidates=[]
    for component,region in enumerate(ndimage.find_objects(labels),1):
        if region is None:continue
        ys,xs=region;area=int(np.count_nonzero(labels[region]==component))
        if area<25 or xs.stop-xs.start<5 or ys.stop-ys.start<5:continue
        if area>30000:continue
        cx=(xs.start+xs.stop-1)//2;cy=(ys.start+ys.stop-1)//2
        candidates.append({'component':component,'bbox_xyxy':[xs.start,ys.start,xs.stop,ys.stop],
            'center_pixel_xy':[cx,cy],'center_xyz_m':cloud[cy,cx].tolist(),
            'area_pixels':area,'height_range_m':[float(cloud[...,2][labels==component].min()-support),float(cloud[...,2][labels==component].max()-support)]})
    candidates.sort(key=lambda c:(c['bbox_xyxy'][0],c['bbox_xyxy'][1]))
    # Geometry is intentionally unlabelled: the model must identify objects and relations.
    for index,c in enumerate(candidates[:24]):c['candidate_id']=f'c{index+1}'
    return candidates[:24],labels,cloud


def reliable_surface_mask(cloud):
    """Conservative 3x3 visible-geometry continuity; no semantic/oracle state.

    The diagonal of the patch's world-space bounding box upper-bounds its
    optical-depth spread for every camera rotation. Stay below the grasp
    validator's 4 cm depth-discontinuity threshold with a 5 mm margin.
    """
    spread_squared = np.zeros(cloud.shape[:2], dtype=float)
    for axis in range(3):
        values = np.where(np.isfinite(cloud[..., axis]), cloud[..., axis], np.inf)
        high = ndimage.maximum_filter(values, size=3, mode="constant", cval=np.inf)
        low = ndimage.minimum_filter(values, size=3, mode="constant", cval=-np.inf)
        with np.errstate(invalid="ignore"):
            spread_squared += (high - low) ** 2
    return np.isfinite(cloud).all(axis=-1) & (spread_squared < .035 ** 2)


def candidate_surface(candidate,labels,cloud,part, *, reliable=False):
    x0,y0,x1,y1=candidate['bbox_xyxy'];cx,cy=candidate['center_pixel_xy']
    mask = labels == candidate['component']
    if reliable:
        mask = mask & reliable_surface_mask(cloud)
        if not np.any(mask):
            raise ValueError("No reliable visible depth patch in this candidate. Inspect a different current view or region; repeating this selection will not repair its depth.")
    if part in {'center','placement_center'}:
        if reliable:
            ys,xs=np.nonzero(mask);i=np.argmin((xs-cx)**2+(ys-cy)**2)
            return int(xs[i]),int(ys[i]),None
        return cx,cy,None
    if part=='top_surface':
        ys,xs=np.nonzero(mask);heights=cloud[ys,xs,2]
        # Aim at a broad horizontal top patch, not its highest lip/edge. A lip
        # grasp can have two pad contacts yet slip during the following transport.
        dy,dx=np.gradient(cloud,axis=(0,1));normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
        flat=(np.abs(normal[ys,xs,2])>.9*length[ys,xs])&(length[ys,xs]>1e-9)
        upper=heights>=np.quantile(heights,.65)-.005
        eligible=flat&upper
        if np.count_nonzero(eligible)>=8:
            bins,counts=np.unique(np.round(heights[eligible]/.002).astype(int),return_counts=True)
            plane=bins[np.argmax(counts)]*.002
            top=eligible&(np.abs(heights-plane)<=.002)
        else:top=heights>=np.quantile(heights,.9)-.003
        indexes=np.nonzero(top)[0];center=np.median(cloud[ys[top],xs[top],:2],axis=0)
        index=indexes[np.argmin(np.linalg.norm(cloud[ys[top],xs[top],:2]-center,axis=-1))]
        return int(xs[index]),int(ys[index]),None
    targets={'top_rim':(cx,y0+3),'bottom_rim':(cx,y1-4),'left_rim':(x0+3,cy),'right_rim':(x1-4,cy)}
    target=targets[part]
    inner=ndimage.binary_erosion(mask,iterations=1)
    ys,xs=np.nonzero(inner if np.any(inner) else mask)
    index=np.argmin((xs-target[0])**2+(ys-target[1])**2);x,y=int(xs[index]),int(ys[index])
    radial=cloud[y,x,:2]-np.asarray(candidate['center_xyz_m'])[:2]
    norm=np.linalg.norm(radial)
    jaw=[float(radial[0]/norm),float(radial[1]/norm),0.] if norm>.005 else None
    return x,y,jaw


def estimate_placement_opening(candidate,labels,cloud):
    """Visible-region footprint and upper edge, not an observed container floor.

    A tall front wall may fully occlude the bottom. Return an explicitly inferred
    drop target, never pretend that the image-box centre measured that bottom.
    """
    points=cloud[labels==candidate['component']]
    lo,hi=np.quantile(points[:,:2],[.03,.97],axis=0)
    if np.max(hi-lo)>.45:raise ValueError('Region too large/merged for a bounded placement estimate')
    return [*map(float,(lo+hi)/2),float(np.quantile(points[:,2],.9))]


def propose_horizontal_bars(depth,fovy,rotation,position,support,robot_position):
    """Unlabelled elongated horizontal surfaces, requested explicitly for handle search.

    Thin supports are often connected to a large cabinet and absent from whole
    object components. Height slabs separate them using only visible RGB-D.
    """
    cloud=world_cloud(depth,fovy,rotation,position)
    if support is None:return [],None,cloud
    normal=np.zeros_like(cloud)
    normal[2:-2,2:-2]=np.cross(cloud[2:-2,4:]-cloud[2:-2,:-4],cloud[4:,2:-2]-cloud[:-4,2:-2])
    length=np.linalg.norm(normal,axis=-1)
    base=(np.abs(normal[...,2])>.5*length)&(length>1e-8)
    base &= (ndimage.maximum_filter(depth,size=3)-ndimage.minimum_filter(depth,size=3))<.025
    base &= np.linalg.norm(cloud[...,:2]-np.asarray(robot_position)[:2],axis=-1)<.8
    found=[]
    for height in np.arange(support+.025,support+.4,.01):
        regions,count=ndimage.label(base&(np.abs(cloud[...,2]-height)<.008))
        for component,region in enumerate(ndimage.find_objects(regions),1):
            if region is None:continue
            yy,xx=np.nonzero(regions[region]==component);ys=yy+region[0].start;xs=xx+region[1].start
            if len(xs)<40:continue
            points=cloud[ys,xs];_,axes=np.linalg.eigh(np.cov(points[:,:2].T))
            widths=np.ptp(points[:,:2]@axes,axis=0)
            if not (.06<widths[-1]<.18 and widths[-1]>4*max(widths[0],.001)):continue
            center=np.median(points,axis=0)
            found.append((len(xs)*widths[-1]/max(widths[0],.003),center,ys,xs))
    labels=np.zeros(depth.shape,dtype=int);candidates=[]
    for _,center,ys,xs in sorted(found,key=lambda x:-x[0]):
        if any(np.linalg.norm(center-np.asarray(c['center_xyz_m']))<.035 for c in candidates):continue
        if np.mean(labels[ys,xs]>0)>.3:continue
        nearest=np.argmin(np.linalg.norm(cloud[ys,xs]-center,axis=-1));cx,cy=int(xs[nearest]),int(ys[nearest])
        prominence=float(np.median(depth[max(0,cy-12):cy+13,max(0,cx-12):cx+13])-depth[cy,cx])
        if prominence<.005:continue  # a flush board edge is not a protruding bar affordance
        component=len(candidates)+1;labels[ys,xs]=component
        candidates.append({'component':component,'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
             'center_pixel_xy':[cx,cy],'center_xyz_m':cloud[cy,cx].tolist(),'area_pixels':len(xs),
             'geometry_type':'horizontal_elongated_surface_not_semantic_handle',
             'local_depth_prominence_m':prominence,
             'height_range_m':[float(cloud[ys,xs,2].min()-support),float(cloud[ys,xs,2].max()-support)]})
        if len(candidates)>=24:break
    candidates.sort(key=lambda c:(c['bbox_xyxy'][0],c['bbox_xyxy'][1]))
    for index,c in enumerate(candidates):c['candidate_id']=f'c{index+1}'
    return candidates,labels,cloud
