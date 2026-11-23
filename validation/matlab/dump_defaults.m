rise_root = getenv('RISE_TOOLBOX'); if isempty(rise_root), rise_root = fullfile(getenv('HOME'),'RISE_toolbox'); end
addpath(rise_root); rise_startup();
meths={'solve','filter','estimate','irf','simulate','forecast','stoch_simul','historical_decomposition','variance_decomposition','compute_steady_state','posterior_simulator','simulation_diagnostics','check_optimum','log_marginal_data_density','print_solution','is_stable_system','resid','theoretical_autocovariances','simulate_nonlinear','filter_initialization','set_solution_to_companion','prior_plots','check_derivatives','optimal_policy_solver_h','solve_alternatives','refresh','load_data'};
fid=fopen('../ref/rise_defaults.tsv','w'); fprintf(fid,'method\toption\tdefault\tcheck\tmessage\n');
for k=1:numel(meths)
  try
    d=feval(meths{k},rise.empty(0));
    if isstruct(d), fn=fieldnames(d); c=cell(numel(fn),4); for q=1:numel(fn), c{q,1}=fn{q}; c{q,2}=d.(fn{q}); c{q,3}=''; c{q,4}=''; end, d=c; end
    for q=1:size(d,1)
      v=d{q,2};
      try
        if ischar(v), sv=['''' v '''']; elseif isa(v,'function_handle'), sv=func2str(v); elseif isnumeric(v)||islogical(v), sv=mat2str(double(v)); elseif iscell(v), sv=['cell ' mat2str(size(v))]; else, sv=class(v); end
      catch, sv=class(v); end
      chk=''; msg='';
      if size(d,2)>=3 && isa(d{q,3},'function_handle'), chk=func2str(d{q,3}); end
      if size(d,2)>=4 && ischar(d{q,4}), msg=d{q,4}; end
      fprintf(fid,'%s\t%s\t%s\t%s\t%s\n',meths{k},d{q,1},sv,chk,msg);
    end
  catch err
    fprintf(fid,'%s\tERROR\t%s\t\t\n',meths{k},strrep(err.message,newline,' '));
  end
end
fclose(fid); disp('OK');
