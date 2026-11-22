const group=document.querySelector('.switch'), img=document.querySelector('#plate-image'), caption=document.querySelector('#plate-title');
const layers={tissue:['Tissue architecture','Illustrative gland-like structures with stained nuclei surrounding pale lumina.'],nuclei:['Nuclear distribution','The same illustrative tissue field with only the nuclei visible.']};
group.hidden=false;
group.addEventListener('click',event=>{
 const button=event.target.closest('[data-layer]');if(!button)return;
 const name=button.dataset.layer;img.src=`./assets/${name}.webp`;img.alt=layers[name][1];caption.textContent=layers[name][0];
 for(const b of group.querySelectorAll('button'))b.setAttribute('aria-pressed',String(b===button));
});
