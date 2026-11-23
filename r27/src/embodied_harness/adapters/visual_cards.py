"""Readable, unlabelled visual references binding geometry IDs to exact RGB crops."""
from PIL import Image, ImageDraw, ImageFont, ImageOps


def candidate_gallery(rgb,candidates):
    columns=min(4,len(candidates));tile_w,tile_h=176,176
    canvas=Image.new('RGB',(columns*tile_w,((len(candidates)+columns-1)//columns)*tile_h),'#e7e9ed')
    draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=23)
    source=Image.fromarray(rgb)
    for index,candidate in enumerate(candidates):
        left=(index%columns)*tile_w;top=(index//columns)*tile_h
        x0,y0,x1,y1=candidate['bbox_xyxy'];bounds=(max(0,x0-3),max(0,y0-3),min(source.width,x1+3),min(source.height,y1+3))
        crop=ImageOps.contain(source.crop(bounds),(tile_w-8,tile_h-36))
        canvas.paste(crop,(left+(tile_w-crop.width)//2,top+32+(tile_h-36-crop.height)//2))
        draw.rectangle((left,top,left+tile_w-1,top+29),fill='#182633')
        draw.text((left+8,top+2),candidate['candidate_id'],font=font,fill='white')
    return canvas
